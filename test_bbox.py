# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/10 下午6:05

import json
import os

import albumentations
import numpy as np

from transforms import Compose
import cv2
from utils.vis import visualize, show_images
from utils.load_annotation import load_bboxes_index_from_json


def test_cls(img_path, transform):
    img = cv2.imread(img_path)
    data = transform(image=img, info='other info')
    cv2.imshow('images', data['image'])
    cv2.waitKey()


def test_bbox(img_path, annotation_path, transform):

    bboxes, category_ids, category_id_to_name = load_bboxes_index_from_json(annotation_path)

    img = cv2.imread(img_path)
    src_img = visualize(img, bboxes, category_ids, category_id_to_name)
    data = transform(image=img, bboxes=bboxes, category_ids=category_ids)
    dst_image = visualize(data['image'], data['bboxes'], category_ids, category_id_to_name)
    return src_img, dst_image

def test_seg(image_path, annotation_path, transform):
    img = cv2.imread(image_path)
    mask = cv2.imread(annotation_path)

    data = dict()
    data["image"] = img
    data["mask"] = mask
    data = transform(**data)

    result = data["image"]*0.7 + data["mask"]*0.3

    cv2.imshow('result', result.astype(np.uint8))
    cv2.waitKey()

    return data["image"], data["mask"]




if __name__ == "__main__":
    jsons_path = "config/file_h.json"
    img_path = "images/bbox_test/test.jpg"
    annotation_path = "images/bbox_test/test.json"

    bboxes, category_ids, category_id_to_name = load_bboxes_index_from_json(annotation_path)

    with open(jsons_path, 'r') as fp:
        cfg = json.load(fp)
    transform = Compose(cfg['train'])
    while True:
        img = cv2.imread(img_path)
        src_img = visualize(img, bboxes, category_ids, category_id_to_name)
        data = transform(image=img, bboxes=bboxes, category_ids=category_ids)
        dst_image = visualize(data['image'], data['bboxes'], category_ids, category_id_to_name)
        cv2.imshow("result", dst_image)
        cv2.waitKey()