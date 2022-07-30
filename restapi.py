"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
import os
import json
# import tools.utils as utils
# import tools.dataset as dataset
# import gdown
# from MORAN_v2.inference import Recognizer


import torch
import base64
import requests
import json
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import cv2
import easyocr
from MIRNet.mirnet.inference import Inferer
app = Flask(__name__)


inferer = Inferer()
# force_reload to recache
# interpreter = tf.lite.Interpreter(model_path="./MIRNet/mirnet_dr.tflite")
inferer.build_model(
    num_rrg=3, num_mrb=2, channels=64,
    weights_path='./MIRNet/low_light_weights_best.h5'
)


# morn = Recognizer('./MORAN_v2/demo.pth')


reader = easyocr.Reader(
    ['en'], model_storage_directory='./ocrmodel/', download_enabled=False)

model = torch.hub.load('./yolov5', 'custom',
                       path='./yolov5/best.pt', source="local")


@app.route('/')
def home():
    return render_template('home.html')


DETECTION_URL = "/img"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((299,299))
        lightIntensityIncreasedImage = inferer.infer_streamlit(img)

        # im_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

        # payload = json.dumps({"image": im_b64})
        # response = requests.post(
        #     'http://192.168.2.103:5000/img', data=payload, headers=headers)
        # data = response.json()
        # print(data.keys())
        # img = data['image']
        # img_bytes = base64.b64decode(img.encode('utf-8'))
        # img = Image.open(io.BytesIO(img_bytes))

        # reduce size=320 for faster inference
        results = model(lightIntensityIncreasedImage, size=640)
        cropped = results.crop()

        # rect_post=morn.preprocess(cropped[0]['im'])
        # output,length=morn.predict(rect_post)
        # rectified_imge=morn.post_process(output,length)

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpen = cv2.filter2D(cropped[0]['im'], -1, sharpen_kernel)
        # cv2.imwrite("blurremoved.jpg", sharpen)
        result = reader.readtext(sharpen, paragraph=False)

        nm = []

        for i in result:
            nm.append(i[1])
            print(i[1])
        ns = '.'.join(nm)

        return ns.replace("\"", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    port = int(os.environ.get('PORT', 3000))
    args = parser.parse_args()

    # debug=True causes Restarting with stat
    app.run(host="0.0.0.0", port=port)
