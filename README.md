# System features

The face recognition system has the following features:
+ Remind users to remove their masks for attendance
+ Detect users using fake faces (Face is displayed on the phone, printed on paper) to deceive the system.

Note: The system works well when the face is < 40 cm from the camera/webcam.



## How to run
Clone repo:
`git clone https://github.com/hisus2000/Attendance-System.git `

Set up environment:
`conda env create -f environment.yml`

Download YOLOv5:
`git clone https://github.com/ultralytics/yolov5.git`

Save the Video containing the employee's face to the link: `./videos`

Run the registration command:
`python reprocessing_and_register.py`

System deployment:
`python main`

## System Accuracy:
Evaluate face mask detection function (Assessed on dataset [Face Mask Dataset](https://drive.google.com/drive/folders/1xllrPRw1Kg1kxbx4dmBMNSahn97G_ZV9?usp=sharing))

Precision Recall: 85% 79%

Evaluate face recognition function (Assessed on LFW dataset):

Acc: 88.05%

## Illustration of the system

The system reminds the user to remove the mask.

![FaceMask](https://github.com/hisus2000/Attendance-System/blob/main/pic/FaceMask.jpg)

User is gotten attendance by the system.

![FaceNoMask](https://github.com/hisus2000/Attendance-System/blob/main/pic/FaceNoMask.jpg)

The system reminds users to use their real face to take attendance.

![FakeFace](https://github.com/hisus2000/Attendance-System/blob/main/pic/FakeFace.jpg)

The system displays the user's attendance information.

![RegisterList](https://github.com/hisus2000/Attendance-System/blob/main/pic/RegisterList.jpg)

The system displays the statistics of the number of late/absent employees.

![Report](https://github.com/hisus2000/Attendance-System/blob/main/pic/Report.jpg)

## References
1. Timesler's facenet repo: [here](https://github.com/timesler/facenet-pytorch)
2. Yolov5_Facemask [here](https://github.com/deepakat002/yolov5_facemask)
3. Yolov5 [here](https://github.com/ultralytics/yolov5)
4. Face Anti Spoofing [here](https://github.com/FaceGg/Face-Anti-Spoofing-RGB)
