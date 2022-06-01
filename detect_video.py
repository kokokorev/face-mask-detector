from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
from pyzbar.pyzbar import decode
import urllib.request, json


def decoder(image):
    # используется BGR формат цвета
    # передаем изображение с камеры в библиотеку комп зрения opencv
    trans_img = cv2.cvtColor(image,0)
    # подготавливаем изображение для определения qr кода (библиотекой pyzbar)
    BarCode = decode(trans_img)

    for obj in BarCode:
        # получаем qr код с изображения
        points = obj.polygon
        (x,y,w,h) = obj.rect
        pts = np.array(points, np.int32)
        # меняем размер изображения qr кода
        pts = pts.reshape((-1, 1, 2))
        
        # ширина обводки
        thickness = 5
        isClosed = True
        
        
        # переводим текст из qr кода в кодировку utf-8
        BarCodeData = obj.data.decode("utf-8")
        
        # просрочен qr код или нет
        is_expired = True
        
        # проверяю, что в qr коде лежит урл госуслуг для проверки сертификата
        if 'https://www.gosuslugi.ru/covid-cert/status/' in str(BarCodeData):
            # достаю qr токен из текста qr кода
            qr_token = str(BarCodeData).split('https://www.gosuslugi.ru/covid-cert/status/')[1].split('?lang=ru')[0]
            # создаю урл к апи с токеном
            url = f'https://www.gosuslugi.ru/api/covid-cert-checker/v3/cert/status/{qr_token}'
            # делаю запрос по урлу к апи
            with urllib.request.urlopen(url) as url:
                # получаю json из ответа по запросу
                data = json.loads(url.read().decode())
                # достаю значение поля isExpired 
                is_expired = data['isExpired']
        
        # настраиваю цвет рамки в зависимости от результата запроса к апи
        color = (0, 255, 0) if is_expired == False else (0, 0, 255)
        cv2.polylines(image, [pts], isClosed, color, thickness)
        
        # настраиваю цвет текста в зависимости от результата запроса к апи
        the_text = "QR Code"
        org = (x,y)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(image, the_text, org, font, 0.9, color, 5)
        
        # вывожу в консоль результат процерки qr кода
        if is_expired == False:
            print('QR Код: Прошел проверку')
        else:
            print('QR Код: Не прошел проверку')


def detect_and_predict_mask(frame, faceNet, maskNet):
    # получаю то что снимает вебка и получаю из нее изображение
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# передаю изображение в нейронку
	faceNet.setInput(blob)
	# detections - результат работы нейронки
	detections = faceNet.forward()
	# print(detections.shape)

	faces = []
	locs = []
	preds = []

	# прохожу по результатм работы нейронки
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		# если высокая вероятность наличия лица
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

		
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			# указываю координаты лица
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# в массив лиц передаю лицо
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	# если найдены лица
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		# ищу маску на лицах
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

# указываю путь к модели по которой будут определяться лица и маски
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")

# запускаю видео
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:
    # указываю размеры окна
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)
	# отправляю окно на опредение qr кода
	decoder(frame)
	# отправляю окно на определение лица и маски
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	# рисую обводку лица
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
  
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# вывожу в консоль результат проверки маски
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