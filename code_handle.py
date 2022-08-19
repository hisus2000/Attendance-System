from time import time
import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from face_recognite import *
from PIL import Image
import models_fas
import imutils
from imutils.video import VideoStream
import os
from gui_handle import *
from datetime import datetime

protoPath = "./face_detector/deploy.prototxt"
modelPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

protoPath2 = "./face_alignment/2_deploy.prototxt"
modelPath2 = "./face_alignment/2_solver_iter_800000.caffemodel"
net2 = cv2.dnn.readNetFromCaffe(protoPath2, modelPath2)

model_name = "MyresNet18"
load_model_path = "./resource/a8.pth"
model = getattr(models_fas, model_name)().eval()
model.load(load_model_path)
model.train(False)

ATTACK=1
# thresh = 0.92# Thresh of FAS
rec_thresh=0.95

thresh =0.92 # Thresh of FAS
# rec_thresh=0.2

# time to reigster
start=1
end=2359

def crop_with_ldmk(image, landmark):
    scale = 3.5
    image_size = 224
    ct_x, std_x = landmark[:, 0].mean(), landmark[:, 0].std()
    ct_y, std_y = landmark[:, 1].mean(), landmark[:, 1].std()

    std_x, std_y = scale * std_x, scale * std_y

    src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
    dst = np.float32([((image_size - 1) / 2.0, (image_size - 1) / 2.0),
                      ((image_size - 1), (image_size - 1)),
                      ((image_size - 1), (image_size - 1) / 2.0)])
    retval = cv2.getAffineTransform(src, dst)
    result = cv2.warpAffine(image, retval, (image_size, image_size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    return result

def demo(img):
    data= np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
    data = data[np.newaxis, :]
    data = torch.FloatTensor(data)
    with torch.no_grad():
        outputs = model(data)
        outputs = torch.softmax(outputs, dim=-1)
        preds = outputs.to('cpu').numpy()
        attack_prob = preds[:, ATTACK]
    return  attack_prob

def fas_rgb(frame,x1,x2,y1,y2):

    (h, w) = frame.shape[:2]

    (startX, startY, endX, endY) = (x1,y1,x2,y2)

    sx = startX
    sy = startY
    ex = endX
    ey = endY

    ww = (endX - startX) // 10
    hh = (endY - startY) // 5

    startX = startX - ww
    startY = startY + hh
    endX = endX + ww
    endY = endY + hh

    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(w, endX)
    endY = min(h, endY)

    x1 = int(startX)
    y1 = int(startY)
    x2 = int(endX)
    y2 = int(endY)

    # roi = frame[y1:y2, x1:x2]
    roi=  frame[startY:endY, startX:endY]

    gary_frame = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    resize_mat = np.float32(gary_frame)
    m = np.zeros((40, 40))
    sd = np.zeros((40, 40))
    mean, std_dev = cv2.meanStdDev(resize_mat, m, sd)
    new_m = mean[0][0]
    new_sd = std_dev[0][0]
    new_frame = (resize_mat - new_m) / (0.000001 + new_sd)
    blob2 = cv2.dnn.blobFromImage(cv2.resize(new_frame, (40, 40)), 1.0, (40, 40), (0, 0, 0))
    net2.setInput(blob2)
    align = net2.forward()

    aligns = []
    alignss = []
    for i in range(0, 68):
        align1 = []
        x = align[0][2 * i] * (x2 - x1) + x1
        y = align[0][2 * i + 1] * (y2 - y1) + y1
        cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)
        align1.append(int(x))
        align1.append(int(y))
        aligns.append(align1)
    cv2.rectangle(frame, (sx, sy), (ex, ey),(0, 0, 255), 2)
    alignss.append(aligns)

    ldmk = np.asarray(alignss, dtype=np.float32)
    ldmk = ldmk[np.argsort(np.std(ldmk[:,:,1],axis=1))[-1]]
    img = crop_with_ldmk(frame,ldmk)
    
    attack_prob = demo(img)
    true_prob = 1 - attack_prob
    if attack_prob > thresh:
        label = 'Fake'                    
    else:
        label = 'Real'
    return label 

def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(700, 500, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)

class live_stream(QThread):

    signal = pyqtSignal(np.ndarray)

    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(live_stream, self).__init__()

    def run(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()  # load model
        self.classes = self.model.names
        # self.input_video = "Input_Video.avi"
        # self.output_video="Output_Video.avi"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.run_program()

    def get_video_from_url(self):
        """
        Creates a new visdeo streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        # /cam
        # return cv2.VideoCapture("giang.mp4")  #"D:/8.Record video/movie.mp4"
        return cv2.VideoCapture(0)  #"D:/8.Record video/movie.mp4"

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model = torch.hub.load('./yolov5', 'custom', path='./resource/last.pt',source='local')
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        global label_face
        global status_fas

        labels, cord = results
        n = len(labels)
        if (n==0):
            f = open("label.txt", "w")
            f.write("")
            f.close()

            f = open("array_label.txt", "w")
            f.write("")
            f.close()
            try:
                os.remove("face_gui.jpg")
            except:
                pass
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]            
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 255, 0)
                face=frame[y1:y2,x1:x2]

                img_gui=cv2.resize(face,(141,141))
                cv2.imwrite("./face_gui.jpg",img_gui) 
                cv2.imwrite("./face.jpg",face) 
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
                
                if self.class_to_label(labels[i])=="Mask":
                    img_fas=frame.copy()
                    cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 3)
                    f = open("label.txt", "w")
                    f.write("Please remove\n your facemask")
                    f.close()
                else:
                    #--------------#
                    # Test casse 1 #
                    #--------------#
                    try:                        
                        try:
                            img_fas=frame.copy()
                            status_fas=fas_rgb(img_fas,x1,x2,y1,y2)                             
                        except:
                            status_fas=""
                        label_face=face_reg(rec_thresh)
                        if status_fas=="Real":
                            f = open("label.txt", "w")
                            f.write(label_face)
                            f.close()
                              
                        else:
                            f = open("label.txt", "w")
                            f.write("Please use real face!")
                            f.close()             
                            label_face="" 

                            f = open("array_label.txt", "w")
                            f.write("")
                            f.close() 
                        # cv2.putText(frame, label_face+" "+status_fas, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                        cv2.putText(frame, label_face, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)                            
                    except:
                        pass                                                     
        return frame

    def run_program(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        player = self.get_video_from_url()
        assert player.isOpened()   
        # # Save Video input
        # x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        # y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # four_cc = cv2.VideoWriter_fourcc(*'XVID')
        # export_input_video = cv2.VideoWriter(self.input_video, four_cc, 20, (x_shape, y_shape))
        # export_output_video = cv2.VideoWriter(self.output_video, four_cc, 20, (x_shape, y_shape))
        while True:
            start_time = time.time()
            ret, frame = player.read()
            assert ret
            # export_input_video.write(frame)
            ####################################################################
            time_regis= int(datetime.now().strftime("%H%M"))
            # Lệnh If này giúp hệ thống hoạt động trong khung giờ cố định
            if not (start< time_regis <end):
                frame=cv2.imread("./resource/time.jpg")
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.time()
            # export_output_video.write(frame)
            fps = 1 / (np.round(end_time - start_time, 3))
            print(f"Frames Per Second : {round(fps,2)} FPS")
            self.signal.emit(frame)
        # export_input_video.release()
        # export_output_video.release()

    def create_report(self):
        data_name=pd.read_csv("data_register.csv")
        data_create=pd.read_csv("database.csv")
        data_process_table=pd.DataFrame(columns=["ID_of_Employee","Time_of_Register","State","Late_Time(minutes)","Delayed","Absence"])
        # Xử lí số lần vắng trong tháng
        absence_people=[]
        late_people=data_name["ID_of_Employee"].values
        for i in (data_create["ID_of_Employee"].values):
            if i not in late_people:
                absence_people.append(i)
        for i in absence_people:
            data_create.loc[data_create['ID_of_Employee'] == i, 'Absence']+=1      
        data_create.to_csv("database.csv",index=False)

    def save_data_report(self):
        ID="01_"
        time=datetime.now().strftime("%d_%B")
        month=datetime.now().strftime("%B")       
        data_register=pd.read_csv("data_register.csv")
        database=pd.read_csv("database.csv")
        name_data_register=ID+"Registered_Staff_Information_"+time
        name_database=ID+"Statistical_List_Of_Employees_In_"+month

        data_register.to_csv(name_data_register+".csv")
        database.to_csv(name_database+".csv")

    def stop(self):
        print("stop threading", self.index)
        os.remove("./face.jpg")
        os.remove("./label.txt")
        os.remove("./array_label.txt")
        os.remove("./face_gui.jpg")
        self.save_data_report()
        self.terminate()
        self.create_report()
        # os.remove("./data_register.csv")