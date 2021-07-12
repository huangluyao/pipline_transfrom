# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/19 下午2:12
import json


def load_bboxes_index_from_json(json_path):
    with open(json_path, 'r') as fp:
        infos = json.load(fp)

    bboxes = []
    labels = []
    for info in infos['shapes']:
        bbox = info['points']
        label = info['label']
        bboxes.append(bbox[0] + bbox[1])
        labels.append(label)

    category_ids = []
    category_id_to_name = dict()

    ids = 0
    for name in labels:
        if name not in category_id_to_name:
            category_id_to_name[name] = ids
            category_ids.append(ids)
            ids += 1
    category_id_to_name = {v: k for k, v in category_id_to_name.items()}

    return bboxes, category_ids, category_id_to_name

