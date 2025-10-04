# =============================================================================
# ULTIMATE VERSION - WITH TRAJECTORIES, ZONES, AND DATA EXPORT
# -----------------------------------------------------------------------------
# This version includes the most advanced features for comprehensive analysis:
# 1.  Trajectory Visualization: Draws the path each object has taken.
# 2.  Danger Zone Monitoring: A configurable zone that triggers alerts.
# 3.  Data Export: Saves the event log to a CSV and the summary to a JSON file.
# 4.  Retains all previous features for maximum accuracy and insight.
# =============================================================================

# =============================================================================
# CELL 1: INSTALL LIBRARIES
# =============================================================================
!pip install -q ultralytics opencv-python

# =============================================================================
# CELL 2: IMPORT LIBRARIES
# =============================================================================
import cv2
from collections import defaultdict, deque
from ultralytics import YOLO
import numpy as np
import os
import logging
import datetime
import csv
import json
import time

# Hide informational messages from ultralytics
logging.getLogger('ultralytics').setLevel(logging.WARNING)
os.environ['YOLO_VERBOSE'] = 'False'


# =============================================================================
# CELL 3: INITIALIZE MODEL AND VIDEO PATHS
# =============================================================================
model = YOLO('yolov8m.pt')

# --- IMPORTANT ---
# Make sure your uploaded video is named 'video1.mp4'
video_path = "video1.mp4"
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file. Please ensure 'video1.mp4' is uploaded correctly.")
except Exception as e:
    print(f"Error: {e}")
    cap = None

if cap:
    track_history = defaultdict(lambda: deque(maxlen=30))
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    still_counters = defaultdict(int)
    # --- Data structures for new features ---
    event_log = []
    previous_tracked_ids = set()
    hazard_logged_ids = set()
    zone_alert_ids = set() # To prevent repeated zone alerts

    output_path = "output_video_ultimate.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    # --- NEW: FRAME SKIPPING FOR FASTER PROCESSING ---
    frame_skip = 1 # Process every frame. Set to 2 to process every other frame, etc.

    # --- NEW: DEFINE DANGER ZONE ---
    # A polygon at the bottom-center of the screen
    danger_zone = np.array([
        [int(frame_width * 0.3), int(frame_height * 0.8)],
        [int(frame_width * 0.7), int(frame_height * 0.8)],
        [frame_width, frame_height],
        [0, frame_height]
    ], np.int32)

    # =============================================================================
    # CELL 4: HELPER FUNCTIONS
    # =============================================================================
    def draw_dashboard(frame, stats, scene_status):
        # This function remains the same
        x, y, w, h = 10, 10, 250, 20 + len(stats) * 20 + 30
        sub_img = frame[y:y+h, x:x+w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.7, white_rect, 0.3, 1.0)
        frame[y:y+h, x:x+w] = res
        cv2.putText(frame, "Scene Analytics", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        row = 0
        for key, value in stats.items():
            text = f"- {key.capitalize()}: {value}"
            cv2.putText(frame, text, (x + 10, y + 45 + row * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            row += 1
        status_color = (0, 128, 0)
        if scene_status == "CAUTION": status_color = (0, 180, 255)
        elif scene_status == "HAZARD": status_color = (0, 0, 255)
        cv2.putText(frame, f"Status: {scene_status}", (x + 10, y + 55 + row * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # =============================================================================
    # CELL 5: MAIN VIDEO PROCESSING LOOP
    # =============================================================================
    object_directions = {}
    
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        if frame_count % frame_skip != 0:
            continue
            
        current_time_sec = frame_count / fps
        
        scene_stats = defaultdict(int)
        scene_status = "SAFE"
        
        # --- NEW: Draw Danger Zone on the frame ---
        cv2.polylines(frame, [danger_zone], isClosed=True, color=(0, 0, 255), thickness=2)
        
        results = model.track(frame, persist=True, verbose=False, tracker="botsort.yaml")

        current_tracked_ids = set()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            xyxys = results[0].boxes.xyxy.cpu()
            
            current_tracked_ids = set(track_ids)

            for box, track_id, cls, xyxy in zip(boxes, track_ids, clss, xyxys):
                track = track_history[track_id]
                center_x, center_y = int(box[0]), int(box[1])
                track.append((float(center_x), float(center_y), float(box[2] * box[3]), frame_count))
                
                direction_text, speed_text = "", ""
                
                if len(track) > 15:
                    # Motion and speed calculations
                    last_5_points = list(track)[-5:]
                    avg_x_last, avg_y_last, avg_area_last, _ = np.mean(last_5_points, axis=0)
                    prev_5_points = list(track)[-10:-5]
                    avg_x_prev, avg_y_prev, avg_area_prev, _ = np.mean(prev_5_points, axis=0)
                    dx, dy, da = avg_x_last - avg_x_prev, avg_y_last - avg_y_prev, avg_area_last - avg_area_prev
                    pos_thresh, area_thresh = 2.5, 150
                    if dx > pos_thresh: direction_text = "Right"
                    elif dx < -pos_thresh: direction_text = "Left"
                    elif dy > pos_thresh: direction_text = "Down"
                    elif dy < -pos_thresh: direction_text = "Up"
                    elif da < -area_thresh: direction_text = "Away"
                    elif da > area_thresh: direction_text = "Towards Ego"
                    else: direction_text = "Still"
                    
                    point_now, point_then = track[-1], track[-10]
                    pixel_dist = np.sqrt((point_now[0] - point_then[0])**2 + (point_now[1] - point_then[1])**2)
                    time_elapsed = (point_now[3] - point_then[3]) / fps
                    speed_px_per_sec = pixel_dist / time_elapsed if time_elapsed > 0 else 0
                    speed_text = f"{int(speed_px_per_sec)} px/s"

                object_class = model.names[cls]
                scene_stats[object_class] += 1
                
                if direction_text == "Still": still_counters[track_id] += 1
                else: still_counters[track_id] = 0
                
                is_vehicle_in_center = object_class in ['car', 'truck', 'bus'] and (frame_width * 0.25 < center_x < frame_width * 0.75)
                if direction_text == "Still" and is_vehicle_in_center and still_counters[track_id] > fps / 2:
                    direction_text = "Pacing"

                if direction_text: object_directions[track_id] = (object_class, direction_text)
                
                color = (0, 255, 0)
                if direction_text == "Towards Ego":
                    color = (0, 255, 255)
                    scene_status = "CAUTION"
                    if 'speed_px_per_sec' in locals() and speed_px_per_sec > (frame_width * 0.1):
                        color = (0, 0, 255)
                        scene_status = "HAZARD"
                        if track_id not in hazard_logged_ids:
                            event_log.append([f"{current_time_sec:.1f}s", "HAZARD", track_id, object_class, "Approaching at high speed."])
                            hazard_logged_ids.add(track_id)

                # --- NEW: Check if object is in the Danger Zone ---
                is_in_zone = cv2.pointPolygonTest(danger_zone, (center_x, center_y), False) >= 0
                if is_in_zone:
                    color = (255, 0, 255) # Purple for danger zone
                    if track_id not in zone_alert_ids:
                        event_log.append([f"{current_time_sec:.1f}s", "DANGER ZONE", track_id, object_class, "Entered critical area."])
                        zone_alert_ids.add(track_id)
                
                # --- NEW: DRAW TRAJECTORY ---
                points = np.array([ [int(p[0]), int(p[1])] for p in track ], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

                p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(frame, p1, p2, color, 2, cv2.LINE_AA)
                label = f"ID:{track_id} {object_class} [{direction_text} | {speed_text}]"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (p1[0], p1[1]-5), (p1[0] + w + 4, p1[1] - h - 10), color, -1)
                cv2.putText(frame, label, (p1[0] + 2, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        
        newly_tracked = current_tracked_ids - previous_tracked_ids
        untracked = previous_tracked_ids - current_tracked_ids
        for tid in newly_tracked:
            event_log.append([f"{current_time_sec:.1f}s", "ENTER", tid, object_directions.get(tid, ('Unknown', ''))[0], "Object appeared."])
        for tid in untracked:
            event_log.append([f"{current_time_sec:.1f}s", "EXIT", tid, object_directions.get(tid, ('Unknown', ''))[0], "Object left scene."])
        previous_tracked_ids = current_tracked_ids

        scene_stats['Total Objects'] = len(current_tracked_ids)
        draw_dashboard(frame, scene_stats, scene_status)

        out.write(frame)
        
        # --- NEW: PROGRESS BAR AND ETR ---
        elapsed_time = time.time() - start_time
        processed_frames = frame_count
        if processed_frames > 0:
            avg_time_per_frame = elapsed_time / processed_frames
            remaining_frames = total_frames - processed_frames
            etr = remaining_frames * avg_time_per_frame
            
            progress = int(50 * processed_frames / total_frames)
            bar = '█' * progress + '-' * (50 - progress)
            print(f'\rProcessing: |{bar}| {processed_frames}/{total_frames} frames. ETR: {etr:.2f}s', end='', flush=True)


    # =============================================================================
    # CELL 6: CLEANUP & FINAL SUMMARY
    # =============================================================================
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    total_processing_time = end_time - start_time
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
    
    print(f"\n\nVideo processing of {frame_count} frames complete.")
    print(f"Output saved to: {output_path}")
    
    # --- NEW: SYSTEM PERFORMANCE MONITOR ---
    print("\n" + "="*30)
    print("      System Performance")
    print("="*30)
    print(f"- Total Processing Time: {total_processing_time:.2f} seconds")
    print(f"- Average FPS: {avg_fps:.2f}")
    print("="*30)

    # --- FINAL SUMMARY (remains the same) ---
    final_object_states = {tid: od for tid, od in object_directions.items() if tid in track_history}
    summary_counts = defaultdict(int)
    summary_movements = defaultdict(lambda: defaultdict(int))
    for name, direction in final_object_states.values():
        summary_counts[name] += 1
        summary_movements[name][direction] += 1

    print("\n" + "="*30)
    print("      Final Video Summary")
    print("="*30)
    print("Total Unique Objects Detected:")
    for name, count in sorted(summary_counts.items()): print(f"- {name.capitalize()}: {count}")
    print("\nTraffic Narrative:")
    for name, movements in sorted(summary_movements.items()):
        movements_str = ", ".join([f"{count} {direction}" for direction, count in sorted(movements.items())])
        print(f"Detected {summary_counts[name]} {name}(s), with final movements: {movements_str}.")
    print("="*30)

    # --- NEW: EXPORT DATA TO FILES ---
    # Export Event Log to CSV
    csv_path = 'events.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'EventType', 'TrackID', 'ObjectClass', 'Details'])
        writer.writerows(event_log)
    print(f"Event log saved to: {csv_path}")

    # Export Final Summary to JSON
    json_path = 'summary.json'
    summary_data = {
        'total_counts': dict(summary_counts),
        'movement_summary': {k: dict(v) for k, v in summary_movements.items()}
    }
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"Final summary saved to: {json_path}")

