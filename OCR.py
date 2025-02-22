import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import os
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

folder_path = './Bang_Diem_Noi_Viet'

model_best = YOLO('best.pt')

image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


for filename in image_files:
    print(filename)


# for filename in image_files[0:2]:
#     image_path = os.path.join(folder_path, filename)

#     results = model_best(image_path)
#     for r in results:

#         img_arr = r.plot(font_size = 3)
#         img = Image.fromarray(img_arr[..., ::-1])

#         if (list(r.boxes.xyxy)[1][1] > list(r.boxes.xyxy)[0][1]):
#             a = 0
#             b = 1
#         else:
#             a = 1
#             b = 0

#         vt1 = list(r.boxes.xyxy)[a]
#         roi1 = img_arr[int(vt1[1] + r.boxes.xywh[a][3]*0.05):int(vt1[3] - r.boxes.xywh[a][3]*0.1), int(vt1[0] + r.boxes.xywh[a][2]*0.012):int(vt1[2] - r.boxes.xywh[a][2]*0.4 )]
#         gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)

#         _, binary_roi1 = cv2.threshold(gray_roi1, 128, 255, cv2.THRESH_BINARY)

#         name = pytesseract.image_to_string(binary_roi1, lang='eng', config='--psm 6')
#         print('Name:', re.sub(r'[^a-zA-Z0-9\s]', '', name))

#         vt2 = list(r.boxes.xyxy)[b]
#         roi2 = img_arr[int(vt2[1] + r.boxes.xywh[b][3]*0.25):int(vt2[3] - r.boxes.xywh[b][3]*0.25), int(vt2[0] +  r.boxes.xywh[b][2]*0.15):int(vt2[2] -  r.boxes.xywh[b][2]*0.15)]
#         gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

#         _, binary_roi2 = cv2.threshold(gray_roi2, 128, 255, cv2.THRESH_BINARY)
#         score = pytesseract.image_to_string(binary_roi2, lang='eng', config='--psm 6')
#         print('Score:', score)

#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
