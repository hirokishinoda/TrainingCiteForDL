import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from model import ConvNet

import base64

from PIL import Image, ImageOps
from io import BytesIO

from flask import Flask
from flask.templating import render_template
from flask import request, jsonify

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

@app.route("/")
def index():
    #default_val = [str(i) for i in range(10)]
    return render_template("index.html", title="手書き文字認識")

@app.route("/predict", methods=['POST'])
def predict():

    #if request.headers['Context-Type'] != 'application/json; charset=utf-8':
    #    return jsonify(res='error'), 400

    # request to image
    json = request.get_json()
    data = json['data'].replace('data:image/png;base64,', '')
    image_binary = base64.b64decode(data)
    image = Image.open(BytesIO(image_binary))

    # background processing
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])

    image = ImageOps.invert(background.convert('L')).resize((28,28))
    image.save('sample.jpg')

    # pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = transform(image).unsqueeze(0)

    # load model
    model = ConvNet().cpu()
    model.load_state_dict(torch.load("./convnet_state.pth", map_location='cpu'))
    model.eval()

    # predict
    output = F.softmax(model(image)).squeeze(0).tolist()
    output = ['{:.2f}'.format(round(val, 4)) for val in output]
    #_, prediction = torch.max(output, 1)

    return jsonify({"predict":output, 'status':200})

if __name__ == '__main__':
    app.run(port=5000, debug=True)