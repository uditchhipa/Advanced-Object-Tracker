Advanced Vehicle & Pedestrian Tracking with YOLOv8
This project is a powerful, real-time video analytics tool that uses the YOLOv8 model to perform advanced object tracking, motion analysis, and threat detection. It processes a video file to identify, track, and analyze the behavior of objects like cars, pedestrians, and more, outputting an annotated video and detailed data logs.

Key Features
This tool is packed with advanced features, making it suitable for a wide range of applications from traffic analysis to security surveillance.

High-Accuracy Object Tracking: Utilizes the powerful YOLOv8m model combined with the BoT-SORT algorithm for robust and precise tracking, even when objects are temporarily obscured.

Comprehensive Motion Analysis: Determines the direction of each object (Left, Right, Towards Ego, Away, Still) and intelligently identifies vehicles that are Pacing with the camera.

Relative Speed Estimation: Calculates and displays the speed of each object in pixels per second, offering quantitative insight into scene dynamics.

Dynamic Hazard Alerts: Bounding boxes change color to indicate the threat level:

GREEN: Normal tracking.

YELLOW: Caution (object is moving towards the camera).

RED: Hazard (object is approaching at high speed).

Trajectory Visualization: Draws the recent path of each object, providing an intuitive visual history of its movement.

Configurable Danger Zone: A designated area on the screen that triggers a special PURPLE alert and a log event when an object enters it.

Real-Time Analytics Dashboard: An on-screen display showing live statistics, including a total object count, a breakdown by class, and the overall scene status (SAFE, CAUTION, HAZARD).

Detailed Event Logging: Records key events with timestamps, including when objects enter/leave the scene and when hazard or danger zone events occur.

Data Export: Automatically saves the final analysis for external use:

summary.json: A structured summary of object counts and movements.

events.csv: A detailed log of all recorded events.

Performance Monitoring: Reports the total processing time and average Frames Per Second (FPS) to gauge system performance.

How to Use
This project is designed to be run easily in a Google Colab notebook, which provides a free GPU.

Prerequisites
A Google Account.

A video file you want to analyze (e.g., video1.mp4).

Steps
Open Google Colab: Navigate to colab.research.google.com and create a New notebook.

Enable GPU: In the notebook menu, go to Runtime -> Change runtime type and select T4 GPU from the dropdown.

Upload Your Video:

Click the folder icon 📁 on the left sidebar.

Click the Upload icon and select your video file.

Important: Ensure the uploaded file is named video1.mp4, or change the video_path variable in the script to match your filename.

Run the Script:

Copy the entire contents of the yolo_tracker.py script into a single code cell in your notebook.

Run the cell by clicking the ▶️ play button or pressing Shift + Enter.

View Results: The script will display a live progress bar. Once complete, the output files (output_video_ultimate.mp4, events.csv, summary.json) will appear in the file browser on the left.

Configuration
You can easily tweak the script's behavior by changing these variables near the top of the code:

video_path: The name of your input video file.

frame_skip: Set to 1 to process every frame, 2 for every other frame, etc., to speed up analysis on long videos.

danger_zone: You can adjust the coordinates of the polygon to change the shape and location of the danger zone.

Output Files
output_video_ultimate.mp4: The processed video with all visualizations (bounding boxes, trajectories, dashboard).

events.csv: A spreadsheet-ready log of all timestamped events.

summary.json: A structured data file containing the final analytics summary.

Dependencies
The script will automatically install the necessary Python libraries:

ultralytics

opencv-python
