from PIL import Image


class T4saSamples(object):
    def __init__(self):
        self.samples = []

    def add_sample(self, image_path, description, classification):
        self.samples.append({
            "image_path": image_path,
            "description": description,
            "classification": classification,
            "image": []
        })

    def get_sample(self, index, load_picture=False):
        if load_picture:
            sample = self.samples[index]
            sample["image"] = Image.open(self.samples["image_path"])
            return sample
        else:
            return self.samples[index]

    def get_samples_size(self):
        return len(self.samples)

