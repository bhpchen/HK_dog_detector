import cv2
import numpy as np 
import yaml
from pytorch_mtcnn_arcface.face_embedding_manager import FaceEmbeddingManager
from PIL import Image
import requests
import datetime
import json
import argparse
import time
import math
from NetworkCameraReader import VideoCapture

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", required=True,
   help="first operand")
args = vars(ap.parse_args())

def load_config(path_to_config):
    try:
        with open(path_to_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.loader.FullLoader)
    except FileNotFoundError:
        config = None
    return config

def initialize_cameras():
    cam_1 = VideoCapture(CAM_USERNAME_1,CAM_PASSWORD_1,CAM_IP_1)
    return cam_1

config = load_config('configuration.yml')


if int(args['class']) == 1:
    CAM_IP_1       = config['CAMERA']['IP']['_1']
    CAM_USERNAME_1 = config['CAMERA']['USERNAME']['_1']
    CAM_PASSWORD_1 = config['CAMERA']['PASSWORD']['_1']
elif int(args['class']) == 2:
    CAM_IP_1       = config['CAMERA']['IP']['_2']
    CAM_USERNAME_1 = config['CAMERA']['USERNAME']['_2']
    CAM_PASSWORD_1 = config['CAMERA']['PASSWORD']['_2']



cam_1 = initialize_cameras()


dogdetector=Detector()


url = "http://localhost:3001/api/photos"
json_result={"result":[]}
headers = {"authorization":"HKPCFRfh&hSECRETadvbbvasds009sdfh32#*FH6f8*Fhsaa"}#,"Content-type":"multipart/form-data"}

def inference(img,camNum):
    global json_result

    detected_boxes=dogdetector.detect(img=img)
    if(detected_boxes):
        for index, box in enumerate(detected_boxes):
            x_1 = int(detected_boxes[0][index][0])
            y_1 = int(detected_boxes[0][index][1])
            x_2 = int(detected_boxes[0][index][2])
            y_2 = int(detected_boxes[0][index][3])

            json_array = [{"identity":identity,"bounding_box":[x_1,y_1,x_2,y_2]}]
            json_result['result'].extend(json_array)
            
        json_result.update({"cam":str(camNum)})
        print(json_result)
        cv2.imwrite('savedImage.jpg',img)
        #cv2.imwrite('savedImage.jpg',img_1)
        
        data = {"json": json.dumps(json_result)}
        files = {"photo": open('savedImage.jpg', 'rb')}
        requests.post(url, data=data,files=files,headers=headers)

def main():

    while True:
        try:
            img_1 = cam_1.read()
            timeStamp = time.time()
            inference(img_1,int(args['class']))

        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()