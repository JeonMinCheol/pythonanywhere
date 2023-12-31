import os
import cv2
import pathlib
from pathlib import Path
import requests
from datetime import datetime
from utils.general import increment_path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
project = ROOT / '.runs/detect/exp',  # save results to project/name
exist_ok = False,  # existing project/name ok, do not increment
save_txt = False,  # save results to *.txt
name = 'exp',  # save results to project/name


class SendData:
    HOST = "http://sdkyu12345.pythonanywhere.com"
    username = "admin"
    password = "my0504"
    title = ''
    text = ''

    def send(self, request, saveName, token):
        now = datetime.now()
        now.isoformat()
        save_dir = increment_path(
            'runs/detect/exp', exist_ok=exist_ok)  # increment run

        self.title = "위반 차량 발견"
        self.text = request["text"]
        image = request["image"]

        today = datetime.now()
        save_path = (os.getcwd() / save_dir / 'detected' /
                     str(today.year) / str(today.month) / str(today.day))
        full_path = save_path / saveName

        dst = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(full_path, dst)

        headers = {'Authorization': 'JWT ' +
                   token, 'Accept': 'application/json'}

        data = {
            'author': 1,
            'title': self.title,
            'text': self.text,
            'created_date': now,
            'published_date': now
        }
        file = {'image': open(full_path, 'rb')}
        res = requests.post(self.HOST + '/api_root/Post/',
                            data=data, files=file, headers=headers)
        print(res)
