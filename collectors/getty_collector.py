from collectors.base_collector import BaseCollector


class GettyCollector(BaseCollector):

    def __init__(self, max_collection, data_root):
        super().__init__(max_collection, data_root)
        self._api_url = "https://developer.gettyimages.com/api/api-overview.html"

    def collect(self):
        pass