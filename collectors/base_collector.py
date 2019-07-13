import os


class BaseCollector:
    def __init__(self, max_collection, data_root):
        self._max_collection = max_collection
        self._data_root = data_root
        self._images_path = os.path.join(self._data_root, "images")
        self._annotations_path = os.path.join(self._data_root, "annotations")
        self._metadata_path = os.path.join(self._data_root, "metadata")

    def collect(self):
        raise NotImplementedError