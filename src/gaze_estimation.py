from model import Model
import cv2

class GazeEstimationModel(Model):
    def __init__(self, model, device = 'CPU', extensions = None):
        Model.__init__(self)

        self.load_model(model, device, extensions)

    def predict(self, image):
        pass

    def preprocess_output(self, outputs):
        pass
