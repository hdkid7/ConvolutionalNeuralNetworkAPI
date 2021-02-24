from flask import Flask, jsonify
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageOps
from Negate import Negative
import io
import torch
from numbernet import Network

# app = Flask(__name__)

model = Network()

model.load_state_dict(torch.load("./mnist"))

model.eval()

# @app.route('/predict', methods=['POST'])
# def predict():
#     return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def transform_img_tensor(img_url):
    my_transforms = transforms.Compose([transforms.Resize(28, 28),
                                        Negative(),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor()

                                        ]
                                       )

    image = Image.open(io.BytesIO(getImageBytes(img_url)))

    return my_transforms(image).unsqueeze(0)


def getImageBytes(path):
    with open(path, 'rb') as f:
        return f.read()


def get_prediction(url):
    tensor = transform_img_tensor(url)
    outputs = model(tensor)
    _, pred = outputs.max(1)
    return pred


print(get_prediction('./draw.png').item())

# if __name__ == '__main__':
#     app.run()
