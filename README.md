Drowsiness Detection System Walkthrough

Prerequisites
Python 3.10 installed (verified)
Webcam connected

How to Run
Open a terminal.
Navigate to the project directory:

cd /Users/rabeehpullichola/drowsiness_detection
Activate the virtual environment:

source venv/bin/activate
Run the application:

python main.py

Verification Steps
Webcam Feed: Ensure the webcam feed appears in a new window.
Face Mesh: Verify that the face mesh (grid of points) is drawn on your face.

Status Labels:
AWAKE: Keep your eyes open and mouth closed. The status should be "AWAKE" (Green).

SLEEPING: Close your eyes. The status should change to "SLEEPING" (Red).

YAWNING: Open your mouth wide. The status should change to "YAWNING" (Yellow).

Values: Check that EAR and MAR values change as you move your facial features.

Troubleshooting
If the camera doesn't open, check if another application is using it.
If the detection is jittery, ensure good lighting.
Press 'q' to quit the application.
