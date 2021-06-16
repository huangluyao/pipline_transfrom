# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/10 下午6:05

import json
from transforms import Compose
import cv2

if __name__ == "__main__":

    json_path = "config/test.json"

    with open(json_path, 'r') as fp:
        cfg = json.load(fp)

    transform = Compose(cfg["train"])

    img = cv2.imread('images/image_0741.jpg')

    data = transform(image=img, info='image')

    cv2.imwrite('images/test.jpg', data['image'])


