class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build model first!")
        print("model saving...")
        self.model.save_weights(checkpoint_path)
        print("model saved!")

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build model first!")
        print("loading model checkpoint {} ... \n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("model saved!")

    def build_model(self):
        raise NotImplementedError



