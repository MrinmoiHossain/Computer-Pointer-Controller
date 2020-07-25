from model import Model
import cv2

class HeadPoseEstimationModel(Model):
    def __init__(self, model, device = 'CPU', extensions = None):
        Model.__init__(self)

        self.load_model(model, device, extensions)

    def predict(self, image):
        self.exec_net(image)

        if self.wait() == 0:
            outputs = self.exec_network.requests[0].outputs
            return outputs

    def preprocess_output(self, outputs, image):
        yaw = outputs['angle_y_fc'][0][0] 
        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]

        cv2.putText(image, "YAW:{:.1f}".format(yaw), (20,20), 0, 0.6, (255,255,0))
        cv2.putText(image, "PITCH:{:.1f}".format(pitch), (20,40), 0, 0.6, (255,255,0))
        cv2.putText(image, "ROLL:{:.1f}".format(roll), (20,60), 0, 0.6, (255,255,0))
        
        return image, [yaw, pitch, roll]
