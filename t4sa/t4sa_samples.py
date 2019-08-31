import torch
from PIL import Image
from torchvision.transforms import transforms

from image_sentiment.vgg19 import settings


class T4saSamples(object):
    def __init__(self):
        self.samples = []
        self._image_preprocess = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize([0, 0, 0], [1, 1, 1], inplace=False)])

    def add_sample(self, image_path, description, classification):
        self.samples.append({
            "image_path": image_path,
            "description": description,
            "classification": classification,
            "image": []
        })

    def get_sample(self, index, load_picture=False, image_size=None):
        sample = self.samples[index]
        sample["classification"] = torch.tensor(int(sample["classification"]))
        if load_picture:
            sample["image"] = self._image_preprocess(self.resize(Image.open(sample["image_path"]), image_size))
            return sample

        return sample

    def get_samples_size(self):
        return len(self.samples)

    def resize(self, im, desired_size):
        old_size = im.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))
        return new_im
