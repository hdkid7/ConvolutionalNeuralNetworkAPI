from flask import Flask, jsonify, request
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageOps
from Negate import Negative
import io
import torch
from numbernet import Network
import json

app = Flask(__name__)

model = Network()

model.load_state_dict(torch.load("./mnist"))

model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(img_bytes)

        return jsonify({'cnn_prediction': class_name.item()})



classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def transform_img_tensor(img_bytes):
    my_transforms = transforms.Compose([transforms.Resize(28, 28),
                                        Negative(),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor()
                                        ]
                                       )

    image = Image.open(io.BytesIO(img_bytes))

    return my_transforms(image).unsqueeze(0)


def getImageBytes(path):
    with open(path, 'rb') as f:
        return f.read()


def get_prediction(img_bytes):
    tensor = transform_img_tensor(img_bytes)

    outputs = model(tensor)

    return outputs.max(1)


if __name__ == '__main__':
    app.run()
