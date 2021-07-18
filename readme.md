# Pipline Transfrom 

This efficient image augmentation pipline, which only depends on OpenCV and numpy.

# Usage examples

1. set up config file in json format.

```json
{
  "train": [
    {"type": "RandomFlip", "prob": 0.5, "direction": "vertical"},
    {"type": "Resize", "height": 400, "width": 400, "always_apply": true},
    {"type": "Rotate", "limit": [-5,5], "prob": 0.5},
    {"type": "ColorJitter","brightness": 0.1, "contrast": 0.5, "saturation": 0.1, "hue": 0.05, "prob": 1}

  ],
  "no use": [
    {"type": "Normalize", "mean": [0.485,0.456,0.406], "std": [0.229,0.224,0.225], "always_apply": true}
  ]
}

```

2. The code is as follows

```python
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

```

## Documentation

Document will be released later and more image augmentation will be added later.



