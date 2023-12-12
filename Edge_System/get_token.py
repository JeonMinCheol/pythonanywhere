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


class token:
    HOST = "http://sdkyu12345.pythonanywhere.com"
    username = "admin"
    password = "my0504"
    token = ''
    title = ''
    text = ''

    def __init__(self):
        res = requests.post(self.HOST + '/api-token-auth/', {
            'username': self.username,
            'password': self.password
        })

        res.raise_for_status()

        self.token = res.json()['token']
