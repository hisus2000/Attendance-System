import cv2 
import uuid
import glob
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm
import dlib
import os
from mtcnn.mtcnn import MTCNN
import glob
import shutil
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import tensorflow as tf
import time

detector = MTCNN()
train_paths = glob.glob("video/*")
df_name_video = []
for i,train_path in tqdm(enumerate(train_paths)):
    name_video = train_path.split("\\")[-1]    
    df_name_video.append(name_video)   
for video in df_name_video:
    name_dic=video.split('.')[0]    
    dic="./images/"+name_dic+"/"     
    try: 
        os.mkdir(dic)        
    except:
        pass 
    img_name=dic       
    counter=0
    i=0 
    cap = cv2.VideoCapture("./video/"+video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            i+=1
            if(i%4==0):
                imgname=img_name+"/"+name_dic+"_0{}.jpg".format(str(counter))
                cv2.imwrite(imgname, frame)
                cv2.imshow('frame', frame)
                counter+=1             
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
#=====================================================================================================
######################################################################################################
#                                
#           Register Face    
#           
######################################################################################################
start_time = time.time()
print("Please waiting for a moment!")
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

dataset=datasets.ImageFolder('images') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = [] # list of cropped faces from photos folder
name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True) 
    if face is not None and prob>0.90: # if face detected and porbability > 90%
        emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
        # emb=tf.math.l2_normalize(emb)
        embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
        name_list.append(idx_to_class[idx]) # names are stored in a list    
meo=embedding_list     
df=pd.DataFrame(columns=["tensor","name"])
df["name"]=name_list
df["tensor"]=embedding_list
number_people=df.groupby('name')["tensor"].count()
tensor=df.groupby('name')["tensor"].sum()
embedding_list=tensor/number_people
name_list=dict.fromkeys(name_list)
name_list=list(name_list) 
# Normalize L2
for i in range (len(embedding_list)):
    embedding_list[i]=torch.nn.functional.normalize(embedding_list[i], p = 2, dim = 1)

data = [embedding_list, name_list]
torch.save(data, 'data.pt') # saving data.pt file
end_time = time.time()
print(f"Estimate for register : {round(end_time - start_time,2)} (s)")
print("It's Done!")
#=====================================================================================================
######################################################################################################
#                                
#           Create file  
#           
######################################################################################################
train_paths = glob.glob("images/*")
df_name_video = []
for i,train_path in tqdm(enumerate(train_paths)):
    name_video = train_path.split("\\")[-1]    
    df_name_video.append(name_video)   
df_name_video=sorted(df_name_video)
declare_data=pd.DataFrame(columns=["ID_of_Employee","Delayed","Absence"])
declare_data["ID_of_Employee"]=df_name_video
declare_data["Delayed"]=declare_data["Absence"]=np.zeros(len(df_name_video))
declare_data["Delayed"]=declare_data["Delayed"].astype("int")
declare_data["Absence"]=declare_data["Absence"].astype("int")
declare_data.to_csv("database.csv",index=False)
data_register=pd.DataFrame(columns=["ID_of_Employee","Time_of_Register","State","Late_Time(minutes)"])
data_register.to_csv("data_register.csv",index=False) 
#=====================================================================================================
