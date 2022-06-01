# import the necessary packages
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from pyzbar.pyzbar import decode
import urllib.request, json


def decoder(image):
    # We're using BGR color format
    trans_img = cv2.cvtColor(image,0)
    BarCode = decode(trans_img)

    for obj in BarCode:
        points = obj.polygon
        (x,y,w,h) = obj.rect
        pts = np.array(points, np.int32)
        # box size
        pts = pts.reshape((-1, 1, 2))
        thickness = 5
        isClosed = True
        
        
        # read qr codes (detect and decode qr codes)
        BarCodeData = obj.data.decode("utf-8")
        BarCodeType = obj.type
        
        is_expired = True
        
        if 'https://www.gosuslugi.ru/covid-cert/status/' in str(BarCodeData):
            qr_token = str(BarCodeData).split('https://www.gosuslugi.ru/covid-cert/status/')[1].split('?lang=ru')[0]
            url = f'https://www.gosuslugi.ru/api/covid-cert-checker/v3/cert/status/{qr_token}'
            with urllib.request.urlopen(url) as url:
                data = json.loads(url.read().decode())
                is_expired = data['isExpired']
        
        # fill color (border)
        color = (0, 255, 0) if is_expired == False else (0, 0, 255)
        cv2.polylines(image, [pts], isClosed, color, thickness)
        the_text = "QR Code"
        
        org = (x,y)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(image, the_text, org, font, 0.9, color, 2)
        if is_expired == False:
            print('QR Код: Прошел проверку')
        else:
            print('QR Код: Не прошел проверку')

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

		
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# print(preds)
	return (locs, preds)

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)
	
	decoder(frame)

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
  
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		if label == 'Mask':
			print('Маска: Прошел проверку')
		else:
			print('Маска: Не прошел проверку')

		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frame, 'Face', (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 5)  

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()