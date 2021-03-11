from cv2 import cv2
from imutils.video import VideoStream
import argparse
import numpy as np
import playsound
from threading import Thread

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="alert.mp3",help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,help="index of webcam on system")
args = vars(ap.parse_args())

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'		#NETWORK CONFIGURATION
weightsPath = 'frozen_inference_graph.pb'						#TRAINED WEIGHTS

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean( (127.5, 127.5, 127.5) )
net.setInputSwapRB(True)

vs = VideoStream(src=args["webcam"]).start()

thres = 0.45
nms_threshold = 0.2
phone_counter = 0

#intro = pyfiglet.figlet_format("Accilert")
#print(intro)

driver = input("Enter the driver name: ")

def sound_alarm(path):
	# pass
	# play an alarm sound
	while ALARM_ON:
		playsound.playsound(path)

while True:

	frame=vs.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	classIds, confs, bbox = net.detect(frame,confThreshold=thres)

	bbox = list(bbox)
	confs = list(np.array(confs).reshape(1,-1)[0])
	confs = list(map(float,confs))
	#print(type(confs[0]))
	#print(confs)

	indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)

	for i in indices:
		i = i[0]
		if classNames[classIds[i][0]-1] == "cell phone":
			phone_counter += 1
		box = bbox[i]
		x,y,w,h = box[0],box[1],box[2],box[3]
		cv2.rectangle(frame, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
		cv2.putText(frame,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),2)
		cv2.putText(frame, "confidence: {:.2f}".format(confs[i]), (box[0]+150,box[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

		if phone_counter >= 10:
			if args["alarm"] != "":
				ALARM_ON = True
				t = Thread(target=sound_alarm,
				args=(args["alarm"],))
				t.deamon = True
				t.start()		
				phone_counter = 0	
				ALARM_ON = False

	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
cv2.destroyAllWindows()
vs.stop()
