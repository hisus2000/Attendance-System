#-------------------------------------------------------------
"""
Recognite by Facenet
"""
import time
from facenet_pytorch import MTCNN, InceptionResnetV1,extract_face
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import glob
import shutil
import numpy as np
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20,post_process=False) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion
convert_tensor = transforms.ToTensor()

def face_match(data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    img = Image.open("./face.jpg")      
    face = convert_tensor(img) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
    emb=torch.nn.functional.normalize(emb, p = 2, dim = 1)  # Normolize l2
    saved_data = torch.load('./resource/data.pt') # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db,p=2).item()
        dist_list.append(dist)
        
    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list)) 

#-------------------------------------------------------------
def face_reg(rec_thresh):
    #----------------------------------------------------
    # try:
    #     result=face_match('data.pt')
    #     name=result[0]
    #     if(result[1]>rec_thresh):
    #         name="unknown"  
    # except:
    #     pass    
    result=face_match('data.pt')
    name=result[0]
    # print(result[1])
    if(result[1]>rec_thresh):
        name="unknown"  
    # ---------------------------------------------------- 
    return name   