from model import Model
import cv2

class FaceDetectionModel(Model):
    def __init__(self, model, device = 'CPU', extensions = None):
        Model.__init__(self)

        self.load_model(model, device, extensions)

    def predict(self, image):
        self.exec_net(image)

        if self.wait() == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_blob]
            return outputs
            # image, detections = self.preprocess_output(outputs, image, threshold)
        
    def preprocess_output(self, outputs, image, threshold = 0.6):
        width = int(image.shape[1])
        height = int(image.shape[0])
        
        faces = []

        for i in range(len(outputs[0][0])):
            box = outputs[0][0][i]
            confidence = box[2]
            if confidence >= threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
                faces.append([xmin, ymin,xmax, ymax])

        return image, faces

