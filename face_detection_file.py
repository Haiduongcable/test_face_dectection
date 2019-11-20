import numpy as np
import argparse
import cv2
#import các thư viện cần thiết
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
#sử dụng argarse path to file image cần detect
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
#path to file Caffe prototxt
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
#path to file Pretrained model Res10
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
# Giới hạn tỉ lệ nhận diện với accuracy = 50%
args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#load model với readNetFromCaffe ( deeplearning Caffe framework)

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
#resize input ảnh với kích thước 300*300, normalize input
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):

	confidence = detections[0, 0, i, 2]
	#trích xuất độ chính xác
	if confidence > args["confidence"]:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# nếu độ chính xác cao hơn ngưỡng accuracy ở input -> vẽ box dectection output
		text = "{:.2f}%".format(confidence * 100)
		# đưa ra độ chính xác bằng cv2.text
		y = startY - 10 if startY - 10 > 10 else startY + 10
		# nếu hình ảnh bị mất một phần khuôn mặt! đẩy vị trí của text 10 đơn vị 
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		# vẽ box
		cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		# vẽ text
cv2.imshow("Output", image)
cv2.waitKey(0)
#chiếu output

			