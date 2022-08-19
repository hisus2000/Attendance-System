import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,QTableWidgetItem,QSizePolicy
from code_handle import live_stream, convert_cv_qt
import pandas as pd
from code_handle import *
from gui import Ui_MainWindow
import cv2
from datetime import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtCore    import *
from PyQt5.QtGui     import *

class MainWindow(QMainWindow):    
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.Button_stop.clicked.connect(self.stop_capture_video)
        self.thread = {}
        self.show_table()
        self.show_table_1()
        #----------------------------------------------------------
        #  this block helps eliminate button_start
        #----------------------------------------------------------
        self.thread[1] = live_stream(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)
        self.uic.Widget.setStyleSheet(
            """
            QTabBar::tab:!selected {
            background: GoldenRod
            }
            """
        )
        self.uic.Button_stop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        day_month="List Of Employee Register "+"( "+datetime.now().strftime("%d %B")+")"
         
        month="Report Of Employee Attendance Statistics In "+datetime.now().strftime("%B")
        self.uic.label_14.setText(month)
        self.uic.label_13.setText(day_month)

    def check_status_register(self):
        f = open("label.txt", "r")
        read_file=f.read()
        f.close()        
        if(read_file=="") or (read_file=="Please use real face!") or (read_file=="Please remove\n your facemask"):
            self.uic.Label_status.setText("")
    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        try:
            self.thread[1].stop()
        except:
            pass    
        self.close()

    def start_capture_video(self):
        self.thread[1] = live_stream(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)

 
    def show_wedcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.show_table()
        self.show_table_1()
        qt_img = convert_cv_qt(cv_img)
        self.uic.Label.setPixmap(qt_img)

        number_of_array=5

        pixmap = QPixmap("face_gui.jpg")
        self.uic.Label_img.setPixmap(pixmap)
        self.check_status_register()

        f = open("label.txt", "r")
        label=f.read()        
        if (label!="Please use real face!") and (label!="Please remove\n your facemask") and (label!="unknown") and (label!=""):
            f = open("array_label.txt", "a")
            f.write(label+",")
            f.close()
        #-----------------------------------------------------------
        """This block helps add ID_of_Employee automatically"""  
        name="" 

        try:
            f = open("array_label.txt", "r")
            read_file=f.read()
            f.close() 
            array_face=list(read_file.split(","))
            add_name=array_face[-number_of_array:-1]

            # name=max(set(add_name), key=add_name.count)
            if len(array_face)>number_of_array:
                name=str(max(set(add_name), key=add_name.count))
                print(add_name)
                print(name)
                self.confirm_name(name)
            else:
                name=""    
                print(add_name)
                print(name)        

            if (label!="Please use real face!") and (label!="Please remove\n your facemask") and (label!="unknown") and (label!=""):
                label="Hello\n" +label 
            if label=="unknown":
                self.uic.Label_status.setText("The face doesn't\n register in the system")            
            self.set_text(label)
            self.show_table()
            self.show_table_1()
        except:
            pass
    def set_text(self,text):
        self.uic.Label_name.setText(text)

    def create_report_real_time(self,data):
        data_name=pd.read_csv("data.csv")
        data_create=pd.read_csv("database.csv")
        data_name["ID_of_Employee"]=data_name["ID_of_Employee"].astype("str")
        data_create["ID_of_Employee"]=data_create["ID_of_Employee"].astype("str")
        data_create["Delayed"]=data_create["Delayed"].astype("int")
        data_create["Absence"]=data_create["Absence"].astype("int")
        # Xử lí số lần trễ trong tháng
        late_people=[]
        late_people=data_name.ID_of_Employee.values[0]
        if (data_name["State"]=="Late")[0]:
            data_create.loc[data_create['ID_of_Employee'] == late_people, 'Delayed']+=1   
        data_create.to_csv("database.csv",index=False)
        
    def confirm_name(self,name_employee):

        # Register  late time
        start=1
        end=800
        state=""
        #-------------------------
        try:
            data_register=pd.read_csv("data_register.csv")
            data_register['ID_of_Employee']=data_register['ID_of_Employee'].astype("str")      
        except:
            data_register=pd.DataFrame(columns=["ID_of_Employee","Time_of_Register","State","Late_Time(minutes)"])
            data_register.to_csv("data_register.csv",index=False) 

        #Kiểm tra xem tên người đó đã điểm danh chưa? Nếu rồi thì không cần add vào cơ sở dữ liệu
        temp=list(data_register["ID_of_Employee"].astype("str"))      
        data_register['ID_of_Employee']=data_register['ID_of_Employee'].astype("str")
        if (name_employee not in temp):            
            # Process        
            now = datetime.now()
            label=name_employee
            time_register=now.strftime("%d %B - %H:%M")  
            time_state_register=int(now.strftime("%H%M"))
            if (start < time_state_register < end):
                state="On Time"
                time_late=0
            else:
                state="Late" 
                time_late=((time_state_register//100)-(end//100))*60+abs(int((time_state_register%100)-(end%100)))                              
            # Add Name to data csv
            data_tempt=pd.DataFrame([[label,time_register,state,time_late]],columns=["ID_of_Employee","Time_of_Register","State","Late_Time(minutes)"])
            data_tempt.to_csv("data.csv",index=False)            
            data_register=pd.concat([data_register,data_tempt])
            data_register=data_register[["ID_of_Employee","Time_of_Register","State","Late_Time(minutes)"]]
            data_register=data_register.reset_index(drop=True)
            data_register.to_csv("data_register.csv",index=False)
            self.create_report_real_time(data_tempt)
            # os.remove("data.csv")

        if (name_employee in temp):            
            self.uic.Label_status.setText("Done Register !")
        else:    
            self.uic.Label_status.setText("")
        self.show_table()
        self.show_table_1()
    def show_table(self):
        try:             
            self.data = pd.read_csv("data_register.csv")
            self.data=self.data[["ID_of_Employee","Time_of_Register","State","Late_Time(minutes)"]]
            numColomn = len(self.data)
            if numColomn == 0:
                NumRows = len(self.data.index)
            else:
                NumRows = numColomn
            self.uic.Table.setColumnCount(len(self.data.columns))
            self.uic.Table.setRowCount(NumRows)
            self.uic.Table.setHorizontalHeaderLabels(self.data.columns)
            
            for i in range(NumRows):
                for j in range(len(self.data.columns)):
                    self.uic.Table.setItem(i, j, QTableWidgetItem(str(self.data.iat[i, j])))
            self.uic.Table.resizeColumnsToContents()
            self.uic.Table.resizeRowsToContents()  
        except:
            pass 

    def show_table_1(self):
        try:             
            self.data_1 = pd.read_csv("database.csv")
            self.data_1=self.data_1[["ID_of_Employee","Delayed","Absence"]]
            numColomn = len(self.data_1)
            if numColomn == 0:
                NumRows = len(self.data_1.index)
            else:
                NumRows = numColomn
            self.uic.Table_1.setColumnCount(len(self.data_1.columns))
            self.uic.Table_1.setRowCount(NumRows)
            self.uic.Table_1.setHorizontalHeaderLabels(self.data_1.columns)
            
            for i in range(NumRows):
                for j in range(len(self.data_1.columns)):
                    self.uic.Table_1.setItem(i, j, QTableWidgetItem(str(self.data_1.iat[i, j])))
            self.uic.Table_1.resizeColumnsToContents()
            self.uic.Table_1.resizeRowsToContents()  
        except:
            pass                    

    def convert_cv_qt(cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(680, 500, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
    self.setFixedSize(self.Widget.sizeHint())