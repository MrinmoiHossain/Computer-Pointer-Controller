from model import Model
import cv2

class GazeEstimationModel(Model):
    def __init__(self, model, device = 'CPU', extensions = None):
        Model.__init__(self)

        self.load_model(model, device, extensions)

    def preprocess_eye(self, frame, face, eye_point):
        eye_input_shape = [1, 3, 60, 60]

        x_center = eye_point[0]
        y_center = eye_point[1]

        width = eye_input_shape[3]
        height = eye_input_shape[2]

        face_width = face.shape[1]
        face_height = face.shape[0]

        xmin = int(x_center - width // 2) if int(x_center - width // 2) >= 0 else 0
        xmax = int(x_center + width // 2) if int(x_center + width // 2) <= face_width else face_width

        ymin = int(y_center - height // 2) if int(y_center - height // 2) >= 0 else 0
        ymax = int(y_center + height // 2) if int(y_center + height // 2) <= face_height else face_height

        eye_image = face[ymin:ymax, xmin:xmax]

        image = cv2.resize(eye_image, (eye_input_shape[3], eye_input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        return image

    def preprocess_input(self, frame, image, left_eye_point, right_eye_point):
        left_eye = self.preprocess_eye(frame, image, left_eye_point)
        right_eye = self.preprocess_eye(frame, image, right_eye_point)

        return frame, left_eye, right_eye

    def exec_net(self, left_eye_image, right_eye_image, headpose_angles, request_id = 0):
        return self.exec_network.start_async(request_id, inputs = {'left_eye_image': left_eye_image, 
                                                                   'right_eye_image': right_eye_image,
                                                                   'head_pose_angles': headpose_angles})

    def predict(self, left_eye_image, right_eye_image, headpose_angles):
        self.exec_net(left_eye_image, right_eye_image, headpose_angles)

        if self.wait() == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_blob]
            return outputs

    def preprocess_output(self, outputs, image):
        x = outputs[0][0]
        y = outputs[0][1]
        z = outputs[0][2]

        return image, [x, y, z]