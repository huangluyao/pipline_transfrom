# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/10 下午6:05

import json
from transforms import Compose
import cv2
from utils.vis import visualize
from utils.load_annotation import load_bboxes_index_from_json


def test_cls(img_path, transform):
    img = cv2.imread(img_path)
    data = transform(image=img, info='other info')
    cv2.imshow('images', data['image'])
    cv2.waitKey()


def test_bbox(img_path, annotation_path, transform):

    bboxes, category_ids, category_id_to_name = load_bboxes_index_from_json(annotation_path)

    img = cv2.imread(img_path)
    data = transform(image=img, bboxes=bboxes, category_ids=category_ids)
    dst_image = visualize(data['image'], data['bboxes'], category_ids, category_id_to_name)

    cv2.imshow('result', dst_image)
    cv2.waitKey()


if __name__ == "__main__":

    json_path = "config/test.json"

    with open(json_path, 'r') as fp:
        cfg = json.load(fp)
    transform = Compose(cfg['train'])

    img_path = "images/test.jpg"
    annotation_path = "images/test.json"
    # test_cls(img_path,  transform)
    while True:
        test_bbox(img_path, annotation_path, transform)

    transform = Compose(cfg["train"])

    img = cv2.imread('images/image_0741.jpg')

    data = transform(image=img, info='image')

    cv2.imwrite('images/test.jpg', data['image'])


