from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pytesseract
import re

app = Flask(__name__, static_url_path='/static')

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
model_best = YOLO('best.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return render_template('index.html', error='No file part')

    files = request.files.getlist('files')

    if not files:
        return render_template('index.html', error='No selected files')

    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            filenames.append(filename)

    if not filenames:
        return render_template('index.html', error='Invalid file type')

    results_list = []
    for filename in filenames:
        filename_detect, image_path, results = process_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        results_list.append({'filename': filename, 'filename_detect': filename_detect, 'results': results})

    return render_template('result.html', results_list=results_list)

def process_image(image_path):
    results = model_best(image_path)
    extracted_data = []

    for r in results:
        if len(r.boxes) == 2:
            img_arr = r.plot(font_size = 3)
            img = Image.fromarray(img_arr[..., ::-1])

            if (list(r.boxes.xyxy)[1][1] > list(r.boxes.xyxy)[0][1]):
                a = 0
                b = 1
            else:
                a = 1
                b = 0

            vt1 = list(r.boxes.xyxy)[a]
            roi1 = img_arr[int(vt1[1] + r.boxes.xywh[a][3]*0.05):int(vt1[3] - r.boxes.xywh[a][3]*0.1), int(vt1[0] + r.boxes.xywh[a][2]*0.012):int(vt1[2] - r.boxes.xywh[a][2]*0.4 )]
            gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)

            _, binary_roi1 = cv2.threshold(gray_roi1, 128, 255, cv2.THRESH_BINARY)

            name = pytesseract.image_to_string(binary_roi1, lang='eng', config='--psm 6')
            extracted_data += [name]
            print('Name:', re.sub(r'[^a-zA-Z0-9\s]', '', name))

            vt2 = list(r.boxes.xyxy)[b]
            roi2 = img_arr[int(vt2[1] + r.boxes.xywh[b][3]*0.25):int(vt2[3] - r.boxes.xywh[b][3]*0.25), int(vt2[0] +  r.boxes.xywh[b][2]*0.15):int(vt2[2] -  r.boxes.xywh[b][2]*0.15)]
            gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

            _, binary_roi2 = cv2.threshold(gray_roi2, 128, 255, cv2.THRESH_BINARY)
            score = pytesseract.image_to_string(binary_roi2, lang='eng', config='--psm 6')
            extracted_data += [score]
            print('Score:', score)
        else:
            if not extracted_data:
                return None, image_path, ["No detection"]
    
    filename_detect = secure_filename(os.path.basename(image_path)[:-4] + '_detect.png')
    file_path_detect = os.path.join(app.config['UPLOAD_FOLDER'], filename_detect)

    cv2.imwrite(file_path_detect, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    return filename_detect, image_path, extracted_data

if __name__ == '__main__':
    app.run(debug=True)
