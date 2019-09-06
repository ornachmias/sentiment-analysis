import torch
from PIL import Image
from torchvision.transforms import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class T4saSamples(object):
    def __init__(self):
        self.samples = []
        self._image_preprocess = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def add_sample(self, image_path, description, classification):
        self.samples.append({
            "image_path": image_path,
            "description": description,
            "classification": classification,
            "image": []
        })

    def get_sample(self, index, load_picture=False, image_size=None):
        sample = self.samples[index]
        raw_class = 0
        if int(sample["classification"]) == 2:
            raw_class = 1

        sample["classification"] = torch.tensor(raw_class)
        if load_picture:
            try:
                sample["image"] = self._image_preprocess(self.resize(Image.open(sample["image_path"]), image_size, sample["image_path"]))
            except ValueError:
                print("Failed to process image. Image path: " + sample["image_path"])
            return sample

        return sample

    def get_samples_size(self):
        return len(self.samples)

    def resize(self, im, desired_size, path):
        try:
            old_size = im.size
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            im = im.resize(new_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(im, ((desired_size - new_size[0]) // 2,
                              (desired_size - new_size[1]) // 2))
        except ValueError:
            print("Failed to process image. Image path: " + path)
            ratio = float(desired_size) / min(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            im = im.resize(new_size, Image.ANTIALIAS)
            new_im = im.crop((0, 0, desired_size, desired_size))

        return new_im
