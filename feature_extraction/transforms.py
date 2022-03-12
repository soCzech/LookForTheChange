import random
import numpy as np
from PIL import Image, ImageEnhance


def crop(img_size, prob=1.):
    w, h = img_size
    dw, dh = random.randint(0, w//6), random.randint(0, h//6)
    if random.random() > prob:
        return lambda x: x
    return lambda img: img.crop((dw, dh, w-w//6+dw, h-h//6+dh))


def rotate(max_angle, prob=0.5):
    angle = random.random() * max_angle * 2 - max_angle
    if random.random() > prob:
        return lambda x: x
    return lambda img: img.rotate(angle)


def flip(prob=0.5):
    if random.random() > prob:
        return lambda x: x
    return lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)


def color(prob=1.):
    if random.random() > prob:
        return lambda x: x
    p = random.random()
    if p < 0.333:
        dx = random.random() - 0.5
        return lambda img: ImageEnhance.Contrast(img).enhance(1 + dx)
    elif p < 0.666:
        dx = random.random() - 0.5
        return lambda img: ImageEnhance.Brightness(img).enhance(1 + dx)
    else:
        dx = random.random() - 0.5
        return lambda img: ImageEnhance.Color(img).enhance(1 + dx)


def get_transform(img_size, identity=False):
    transforms = [] if identity else [rotate(5), flip(), crop(img_size), color(prob=0.8)]

    def fc(imgs, end_size=None):
        imgs = [Image.fromarray(img) for img in imgs]
        for t in transforms:
            imgs = [t(img) for img in imgs]
        if end_size is not None:
            imgs = [img.resize(end_size) for img in imgs]
        return np.stack([np.array(img) for img in imgs], 0)
    return fc
