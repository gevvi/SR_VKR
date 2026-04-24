from PIL import Image


class ImagePreprocessor:
    def load(self, path):
        return Image.open(path)

    def preprocess(self, image: Image.Image, convert_to_rgb: bool = True):
        if convert_to_rgb and image.mode != 'RGB':
            image = image.convert('RGB')
        return image
