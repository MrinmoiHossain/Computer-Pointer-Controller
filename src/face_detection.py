from model import Model

class FaceDetectionModel(Model):
    def __init__(self, model, device = 'CPU', extensions = None):
        Model.__init__()

        self.load_model(model, device, extensions)

    def predict(self, image):
        image = self.preprocess_input(image)
        self.exec_net(image)

        if self.wait() == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_blob]
            detections, crop_image = self.preprocess_output(outputs, image)
        
    def preprocess_output(self, outputs, image):
        width = int(image.shape[1])
        height = int(image.shape[0])
        ## TODO