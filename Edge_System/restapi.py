"""
edge server
데이터 추론을 여기서 하고 위반 차량이 발견되면 장고 서버로 데이터 전송
"""
import argparse
from datetime import datetime
import io
import shutil
from flask import Flask, request
from PIL import Image
import torch
import numpy as np
import cv2
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator
from sendData import SendData
from pathlib import Path
from utils.general import (increment_path, non_max_suppression)
import os
import sys
import re



app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5s"

FILE = Path(__file__).resolve()

ROOT = FILE.parents[0]  # YOLOv5 root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
project= '/runs/detect/exp',  # save results to project/name
exist_ok=False,  # existing project/name ok, do not increment
save_txt=False,  # save results to *.txt
name='exp'  # save results to project/name

# 횡단보도,신호등 모델
MODEL_PATH = 'runs/train/exp4/weights/best.pt'

img_size = 640
conf_thres = 0.5  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000  # maximum detections per image
classes = None  # filter by class
agnostic_nms = False  # class-agnostic NMS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ckpt = torch.load(MODEL_PATH, map_location=device)
model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
class_names = ['횡단보도', '빨간불', '초록불']  # model.names
stride = int(model.stride.max())
colors = ((50, 50, 50), (0, 0, 255), (0, 255, 0))  # (gray, red, green)

# 차량,번호판 모델
CONFIDENCE = 0.5
THRESHOLD = 0.3
LABELS = ['차량', '번호판']

net = cv2.dnn.readNetFromDarknet('models/yolov4-ANPR.cfg', 'models/yolov4-ANPR.weights')

def clean_filename(filename):
    # 허용되는 문자와 공백을 언더스코어(_)로 대체합니다.
    cleaned_filename = re.sub(r'[\\/:"*?<>|]', '_', filename)
    cleaned_filename = cleaned_filename.replace(' ', '_')
    return cleaned_filename

def inference(data, saveName, savePath):
    img = cv2.imread(data)
    H, W, _ = img.shape
    cw_x1, cw_x2 = -9999, -9999
    flag = False

    # preprocess
    img_input = letterbox(img, img_size, stride=stride)[0]  # letterbox(img)[0]  # , img_size, stride=stride)[0]
    img_input = img_input.transpose((2, 0, 1))[::-1]
    img_input = np.ascontiguousarray(img_input)
    img_input = torch.from_numpy(img_input).to(device)
    img_input = img_input.float()
    img_input /= 255.
    img_input = img_input.unsqueeze(0)

    # inference
    pred = model(img_input, augment=False, visualize=False)[0]

    # postprocess
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

    pred = pred.cpu().numpy()

    pred[:, :4] = scale_coords(img_input.shape[2:], pred[:, :4], img.shape).round()

    # inference 차량,번호판
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255., size=(416, 416), swapRB=True)
    net.setInput(blob)
    output = net.forward()

    boxes, confidences, class_ids = [], [], []

    for det in output:
        box = det[:4]
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONFIDENCE:
            cx, cy, w, h = box * np.array([W, H, W, H])
            x = cx - (w / 2)
            y = cy - (h / 2)

            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    annotator = Annotator(img.copy(), line_width=3, example=str(class_names), font='data/malgun.ttf')

    for p in pred:
        class_name = class_names[int(p[5])]
        x1, y1, x2, y2 = p[:4]

        # annotator.box_label([x1, y1, x2, y2], '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])
        if class_name == '횡단보도':
            cw_x1, cw_x2 = x1, x2

    if len(idxs) > 0:
        for i in idxs.flatten():
            class_name = LABELS[class_ids[i]]
            x1, y1, w, h = boxes[i]

        alert_text = ''
        color = (255, 0, 0)  # blue

        if x1 < cw_x2:  # 차량의 좌측(x1) 좌표가 횡단보도의 우측(cw_x2)을 침범하였을 경우
            alert_text = '[!위반]'
            flag = True
            color = (0, 0, 255)  # red

        cw_x1, cw_x2 = -9999, -9999
        print(alert_text + class_name)

        annotator.box_label([x1, y1, x1 + w, y1 + h], '%s %d' %
                            (alert_text + class_name, confidences[i] * 100), color=color)

    if flag:
        result_img = annotator.result()
        cv2.imwrite(saveName, result_img)
        shutil.move(saveName, savePath)   
        
        return (True, result_img)
    
    return (False, None)


# request는 title, location, image를 가져야함

@app.route(DETECTION_URL, methods=["POST"])

def predict():
    if not request.method == "POST":
        return
    today = datetime.now()
    
    # Directories
    save_dir = increment_path('./runs/detect/exp', exist_ok=exist_ok)  # increment run
    save_path = (save_dir / 'detected' /str(today.year) / str(today.month) / str(today.day))
    save_path.mkdir(parents=True, exist_ok=True)  # make dir
    full_path = './runs/detect/exp/detected/{0}-{1}-{2}-{3}.jpg'.format(today.hour,today.minute,today.second,today.microsecond)

    sd = SendData()
    
    image_file = request.files["image"]
    image_bytes = image_file.read()
    
    data = {
        "title" : request.form["title"],
        "text" : request.form["text"],
        "image" : None
    }
     
    img = Image.open(io.BytesIO(image_bytes))
    img.save("receiveData.jpg", "JPEG")  # 현재 환경에 save.jpg라는 이름으로 저장
    
    saveName = '{0}-{1}-{2}-{3}.jpg'.format(today.hour,today.minute,today.second,today.microsecond)
    im0 = inference("receiveData.jpg", saveName, save_path)

    if im0[0] == True:
        data["image"] = im0[1]
        sd.send(data, saveName)
    return "."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
