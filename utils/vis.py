import cv2
import matplotlib.pyplot as plt
import numpy as np
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    return img


def show_images(images, titles=None, num_cols=None, save_name=None, scale=3, normalize=False):
    """ 一个窗口中绘制多张图像:
    Args:
        images: 可以为一张图像(不要放在列表中)，也可以为一个图像列表
        titles: 图像对应标题、
        num_cols: 每行最多显示多少张图像
        scale: 用于调整图窗大小
        normalize: 显示灰度图时是否进行灰度归一化
    """

    # 加了下面2行后可以显示中文标题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 单张图片显示
    if not isinstance(images, list):
        if not isinstance(scale, tuple):
            scale = (scale, scale * 1.5)

        plt.figure(figsize=(scale[1], scale[0]))
        img = images
        if len(img.shape) == 3:
            # opencv库中函数生成的图像为BGR通道，需要转换一下
            plt.imshow(img[:,:,::-1])
        elif len(img.shape) == 2:
            # pyplot显示灰度需要加一个参数
            if normalize:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            raise TypeError("Invalid shape " +
                            str(img.shape) + " of image data")
        if titles is not None:
            plt.title(titles, y=-0.15)
        plt.axis('off')
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()
        return

    # 多张图片显示
    if not isinstance(scale, tuple):
        scale = (scale, scale)

    num_imgs = len(images)
    if num_cols is None:
        num_cols = int(np.ceil((np.sqrt(num_imgs))))
    num_rows = (num_imgs - 1) // num_cols + 1

    idx = list(range(num_imgs))
    _, figs = plt.subplots(num_rows, num_cols,
                           figsize=(scale[1] * num_cols, scale[0] * num_rows), constrained_layout=True)
    for f, i, img in zip(figs.flat, idx, images):
        if len(img.shape) == 3:
            # opencv库中函数生成的图像为BGR通道，需要转换一下
            f.imshow(img[:,:,::-1])
        elif len(img.shape) == 2:
            # pyplot显示灰度需要加一个参数
            if normalize:
                f.imshow(img, cmap='gray')
            else:
                f.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            raise TypeError("Invalid shape " +
                            str(img.shape) + " of image data")
        if titles is not None:
            f.set_title(titles[i], y=-0.15)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    # 将不显示图像的fig移除，不然会显示多余的窗口
    if len(figs.shape) == 1:
        figs = figs.reshape(-1, figs.shape[0])
    for i in range(num_rows * num_cols - num_imgs):
        figs[num_rows - 1, num_imgs % num_cols + i].remove()
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
