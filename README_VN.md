# Tính năng của hệ thống

Hệ thống điểm danh bằng nhận diện gương mặt có các tính năng:
+ Nhắc nhở người dùng gỡ khẩu trang để điểm danh
+ Phát hiện người dùng sử dụng gương mặt giả (Gương mặt được hiển thị trên điện thoại, in trên giấy) để đánh lừa hệ thống.

Lưu ý: Hệ thống hoạt động tốt khi gương mặt cách camera/webcam < 40 cm.



## Cách chạy
Clone repo:
`git clonehttps://github.com/hisus2000/Attendance-System.git `

Setup môi trường:
`conda env create -f environment.yml`

Download YOLOv5:
`git clone https://github.com/ultralytics/yolov5.git`

Lưu Video chứa gương mặt nhân viên vào đường dẫn: `./videos`

Chạy đoạn lệnh đăng kí:
`python reprocessing_and_register.py`

Triển khai hệ thống:
`python main`

## Độ chính xác của hệ thống:
Chức năng phát hiện gương mặt mang khẩu trang (Đánh giá trên tập dữ liệu [Face Mask Dataset](https://drive.google.com/drive/folders/1xllrPRw1Kg1kxbx4dmBMNSahn97G_ZV9?usp=sharing))

Precision Recall: 85% 79%

Chức năng nhận diện gương mặt (Đánh giá trên tập dữ liệu LFW):

Acc: 88.05%

## Hình ảnh minh họa Demo của hệ thống

Hệ thống nhắc nhở người dùng gỡ khẩu trang

![FaceMask](https://github.com/hisus2000/Attendance-System/blob/main/pic/FaceMask.jpg)

Hệ thống điểm danh người dùng

![FaceNoMask](https://github.com/hisus2000/Attendance-System/blob/main/pic/FaceNoMask.jpg)

Hệ thống nhắc nhở người dùng sử dụng gương mặt thật để điểm danh

![FakeFace](https://github.com/hisus2000/Attendance-System/blob/main/pic/FakeFace.jpg)

Hệ thống hiển thị thông tin điểm danh của người dùng

![RegisterList](https://github.com/hisus2000/Attendance-System/blob/main/pic/RegisterList.jpg)

Hệ thống hiển thị thống kê số lần đi trễ/vắng mặt của nhân viên

![Report](https://github.com/hisus2000/Attendance-System/blob/main/pic/Report.jpg)

## References
1. Timesler's facenet repo:  [here](https://github.com/timesler/facenet-pytorch)
2. Yolov5_Facemask [here](https://github.com/deepakat002/yolov5_facemask)
3.  Yolov5 [here](https://github.com/ultralytics/yolov5)
4. Face Anti Spoofing [here](https://github.com/FaceGg/Face-Anti-Spoofing-RGB)
