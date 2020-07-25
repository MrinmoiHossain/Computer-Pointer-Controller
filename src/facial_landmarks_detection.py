from model import Model
import cv2

class FacialLandmarksDetectionModel(Model):
    def __init__(self, model, device = 'CPU', extensions = None):
        Model.__init__(self)

        self.load_model(model, device, extensions)

    def predict(self, image):
        self.exec_net(image)

        if self.wait() == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_blob]
            return outputs

    def preprocess_output(self, outputs, image, face):
        normed_landmarks = outputs.reshape(1, 10)[0]

        height = face[3] - face[1]
        width = face[2] - face[0]

        for i in range(2):
            x = int(normed_landmarks[i * 2] * width)
            y = int(normed_landmarks[i * 2 + 1] * height)

            cv2.circle(image, (face[0] + x, face[1] + y), 30, (0, 255, i * 255), 2)

        left_eye_point = [normed_landmarks[0] * width, normed_landmarks[1] * height]
        right_eye_point = [normed_landmarks[2] * width, normed_landmarks[3] * height]

        return image, left_eye_point, right_eye_point
