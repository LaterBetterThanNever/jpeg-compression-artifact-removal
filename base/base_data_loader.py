class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def generate_train_data(self):
        raise NotImplementedError

    def generate_test_data(self):
        raise NotImplementedError
