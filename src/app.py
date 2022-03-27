import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import sys
import os

sys.path.append(os.pardir)
from models import CNN
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
    return render_template("index.html", title="手書き文字認識")

@app.route("/predict", methods=['POST'])
def predict():
    # convert from request to image
    json = request.get_json()
    data = json['data'].replace('data:image/png;base64,', '')
    image_binary = base64.b64decode(data)
    image = Image.open(BytesIO(image_binary))

    # background processing
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])

    image = ImageOps.invert(background.convert('L')).resize((28,28))

    # pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)

    # load model
    model = CNN().cpu()
    model.load_state_dict(torch.load("./src/convnet_state.pth", map_location='cpu'))
    model.eval()

    # predict
    output = F.softmax(model(image)).squeeze(0).tolist()
    output = ['{:.2f}'.format(round(val, 4)) for val in output]

    return jsonify({"predict":output, 'status':200})

def create_app():
    app.run()

if __name__ == '__main__':
    #create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)