import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import threading
from roboflow import Roboflow

rf = Roboflow(api_key = "iNj3iWGu5PZD3njmYV7R")
project = rf.workspace().project("weed-crop-detect")
model = project.version(2).model

GPIO.setmode(GPIO.BOARD)

pump1_pin = 13
pump2_pin = 11

m1 = 15
m2 = 16
m3 = 18
m4 = 22
trig = 24
echo = 26

GPIO.setup(pump1_pin, GPIO.OUT)
GPIO.setup(pump2_pin, GPIO.OUT)
GPIO.setup(trig, GPIO.OUT)
GPIO.setup(echo, GPIO.IN)
GPIO.setup(m1,GPIO.OUT)
GPIO.setup(m2,GPIO.OUT)
GPIO.setup(m3,GPIO.OUT)
GPIO.setup(m4,GPIO.OUT)
GPIO.output(pump1_pin, GPIO.HIGH)
GPIO.output(pump2_pin, GPIO.HIGH)

def forward():
  GPIO.output(m1,GPIO.LOW)
  GPIO.output(m2,GPIO.HIGH)
  GPIO.output(m3,GPIO.LOW)
  GPIO.output(m4,GPIO.HIGH)
  time.sleep(0.1)
  stop()

def backward():
  GPIO.output(m1,GPIO.HIGH)
  GPIO.output(m2,GPIO.LOW)
  GPIO.output(m3,GPIO.HIGH)
  GPIO.output(m4,GPIO.LOW)

def stop():
  GPIO.output(m1,GPIO.HIGH)
  GPIO.output(m2,GPIO.HIGH)
  GPIO.output(m3,GPIO.HIGH)
  GPIO.output(m4,GPIO.HIGH)


def distance():
     while True:
        GPIO.output(trig,True)
        time.sleep(0.00001)
        GPIO.output(trig,False)	
        startTime  = time.time()
        stopTime = time.time()
        while GPIO.input(echo)==0:
           startTime = time.time()
        while GPIO.input(echo)==1:
           stopTime = time.time()
        timeElapsed = stopTime - startTime
        distance = (timeElapsed*34300)/2
        #print("Distance:",distance)
        if distance < 10:
          stop()
          time.sleep(1)
        else:
          forward()
        time.sleep(1)

LABELS = ['crop','weed']
weightsPath = 'crop_weed_detection.weights'
configPath = 'crop_weed.cfg'

print("[INFO] loading YOLO from disk...")
#net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

confi = 0.5
thresh = 0.5
#ln = net.getLayerNames()
#ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
"""cap = cv2.VideoCapture(1)"""

t1 = threading.Thread(target = distance)
t1.start()
count = 0
while True:
    #forward()
    stop()
    time.sleep(1)
    try:
     ret, image = cap.read()
     if ret:
        count = count+1
       # (H, W) = image.shape[:2]
       # blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
       # net.setInput(blob)
       # layerOutputs = net.forward(ln)
       # boxes = []
       # confidences = []
       # classIDs = []
       # for output in layerOutputs:
       #     for detection in output:
       #         scores = detection[5:]
       #         classID = np.argmax(scores)
       #         confidence = scores[classID]
       #         if confidence > confi:
       #             box = detection[0:4] * np.array([W, H, W, H])
       #             (centerX, centerY, width, height) = box.astype("int")
       #             x = int(centerX - (width / 2))
       #             y = int(centerY - (height / 2))
       #             boxes.append([x, y, int(width), int(height)])
       #             confidences.append(float(confidence))
       #             classIDs.append(classID)
       # idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
        res = model.predict(image,confidence=40,overlap=30).json()
        predictions = res['predictions']
        out = predictions[0]
        cls = out['class']
       
        #if len(idxs) > 0:
         #   for i in idxs.flatten():
          #      (x, y) = (boxes[i][0], boxes[i][1])
           #     (w, h) = (boxes[i][2], boxes[i][3])
               
            #    if x < W / 2:
        print("Predicted ->  :  ", cls)
        if cls == 'weed':
           if count % 8 < 2:
             stop()
             GPIO.output(pump1_pin,GPIO.LOW)
             GPIO.output(pump2_pin,GPIO.HIGH)
             time.sleep(2)
             GPIO.output(pump1_pin,GPIO.HIGH)
		
                #else:
                 #   print("Predicted ->  :  ", LABELS[classIDs[i]], "Left")
                  #  if LABELS[classIDs[i]] == 'weed':
                   #     stop()
                    #    GPIO.output(pump1_pin,GPIO.HIGH)
                     #   GPIO.output(pump2_pin,GPIO.LOW)
                      #  time.sleep(3)
                       # GPIO.output(pump2_pin,GPIO.HIGH)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    except IndexError:
      print('No class identified')
      count = 0

cap.release()
cv2.destroyAllWindows()
