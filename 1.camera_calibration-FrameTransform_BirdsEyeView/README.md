The code is based on camera_calibration code found in Github
# Calibrate camera with arUco markers using opencv and python. 
(Code developed and tested on opencv 3.3.1)


# camera_calibration
Code and resources for camera calibration using arUco markers and opencv 

1. Print the aruco marker board provided. (You can generate your own board, see the code "camera_calibration.py")
2. Take around 50 images of the printed board pasted on a flat card-board, from different angles. Use Use data_generation node for this.
3. Set path to store images first.
(See sample images of arUco board in aruco_data folder)

## Calibrating camera
1. Measure length of the side of individual marker and spacing between two marker.
2. Input above data (length and spacing) in "camera_calibration.py".
3. Set path to stored images of aruco marker.
4. Set calibrate_camera flag to true.

camera_calibration.py has been modified for detecting the aruco markers found in scenes. 
They have been used along with birds eye view and frame transforms to estimate the volume.
The yolov3 was used to detect the food item but then countour detection was used instead because 
otherwise the model would have to trained on all food items to detect them (which was beyound the
scope of this project)