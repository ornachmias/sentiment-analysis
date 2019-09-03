import errno
import os

from tqdm import tqdm
from torch.utils.data import Dataset

from t4sa_samples import T4saSamples
import csv


class T4saDataset(Dataset):
    def __init__(self, train, configs, limit=None, load_image=False):
        self._image_size = configs.image_size
        self._load_image = load_image
        self._images_directory = configs.images_directory
        if not os.path.exists(self._images_directory):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.realpath(self._images_directory))

        self._images_classification_file = ""

        if train:
            self._images_classification_file = configs.images_classification_train_file
        else:
            self._images_classification_file = configs.images_classification_eval_file

        if not os.path.exists(self._images_classification_file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self._images_classification_file)

        self._descriptions_file = configs.descriptions_file
        if not os.path.exists(self._descriptions_file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self._descriptions_file)

        self._samples = T4saSamples()
        self._load_samples(limit)

    def __len__(self):
        return self._samples.get_samples_size()

    def __getitem__(self, idx):
        return self._samples.get_sample(idx, self._load_image, self._image_size)

    def _load_samples(self, limit):
        image_id_to_text = self._load_descriptors()
        with open(self._images_classification_file) as classification_file:
            csv_reader = csv.reader(classification_file, delimiter=' ')
            for row in tqdm(csv_reader):
                image_id = os.path.splitext(os.path.basename(row[0]))[0].rsplit("-", 1)[0]
                self._samples.add_sample(os.path.join(self._images_directory, row[0]), image_id_to_text[image_id], row[1])
                if limit is not None and self._samples.get_samples_size() == limit:
                    break

    def _load_descriptors(self):
        image_id_to_text = {}
        with open(self._descriptions_file, encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)
            for row in tqdm(csv_reader):
                image_id_to_text[row[0]] = row[1]

        return image_id_to_text
