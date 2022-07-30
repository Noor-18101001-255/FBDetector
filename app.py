import io 
import cv2 
import numpy as np
from PIL import Image
from flask import Flask,request
from keras.models import load_model
from tensorflow.keras import optimizers


app = Flask(__name__)
# model = load_model('D:\FYP\FYP ML Stuff\Model\FBD90%.h5') 
# # model = model_from_json('model.json') 
# # model.summary()
# model.compile(loss='binary_crossentropy',
# optimizer=optimizers.RMSprop(learning_rate=1e-5),
#               metrics=['accuracy'])

# print(model.summary())

print("Models Ready!")
@app.route('/')
def home():
  return '<h1>Welcome</h1>'

# DETECTION_URL = "/img"
# @app.route(DETECTION_URL, methods=["POST"])
# def predict():
#     if not request.method == "POST":
#         return

#     if request.files.get("image"):
#         image_file = request.files["image"]
#         image_bytes = image_file.read()
#         img = Image.open(io.BytesIO(image_bytes))
#         img = img.resize((299,299))



        # lightIntensityIncreasedImage = inferer.infer_streamlit(img)

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
        # results = model(lightIntensityIncreasedImage, size=640)
        # cropped = results.crop()

        # rect_post=morn.preprocess(cropped[0]['im'])
        # output,length=morn.predict(rect_post)
        # rectified_imge=morn.post_process(output,length)

        # sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # sharpen = cv2.filter2D(cropped[0]['im'], -1, sharpen_kernel)
        # cv2.imwrite("blurremoved.jpg", sharpen)
        # result = reader.readtext(sharpen, paragraph=False)

        # nm = []

        # for i in result:
        #     nm.append(i[1])
        #     print(i[1])
        # ns = '.'.join(nm)

        # return ns.replace("\"", "")




if __name__ == '__main__':
  app.run()
  pass
