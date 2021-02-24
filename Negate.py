import PIL.ImageOps


def to_negative(img):
    img = PIL.ImageOps.invert(img.convert('RGB'))
    return img


class Negative(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return to_negative(img)
