import os
import time
import cv2
import logging
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from gaze_estimation import GazeEstimationModel

def build_argparser():
    parser = ArgumentParser()

    parser.add_argument("-mfd", "--faceDetectionModel", type=str, required=True,
                        help="Path of xml file of face detection model")

    parser.add_argument("-mfl", "--facialLandmarksModel", type=str, required=True,
                        help="Path of xml file of facial landmarks detection model")

    parser.add_argument("-mhp", "--headPoseModel", type=str, required=True,
                        help="Path of xml file of head pose estimation model")

    parser.add_argument("-mge", "--gazeModel", type=str, required=True,
                        help="Path of xml file of gaze estimation model")

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to image or video file")

    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")

    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable.")

    parser.add_argument("-pt", "--prob_threshold", required=False, type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")

    return parser
    
def logfile_config():
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s",
                        handlers = [logging.FileHandler("GazeApp.log"), logging.StreamHandler()])

def infer_on_stream(args):
    face_detection_model_file = args.faceDetectionModel
    facial_landmarks_detection_model_file = args.facialLandmarksModel
    head_pose_estimation_model_file = args.headPoseModel
    gaze_estimation_model_file = args.gazeModel

    video_file = args.input
    device_name = args.device
    cpu_extension = args.cpu_extension
    prob_threshold = args.prob_threshold

    logging.info("*********** Model Load Time ***************")
    start_time = time.time()
    face_detection_model = FaceDetectionModel(face_detection_model_file, device_name, cpu_extension)
    logging.info("Face Detection Model: {:.1f} ms.".format(1000 * (time.time() - start_time)))

    start_time = time.time()
    facial_landmarks_detection_model = FacialLandmarksDetectionModel(facial_landmarks_detection_model_file, device_name, cpu_extension)
    logging.info("Facial Landmarks Detection Model: {:.1f} ms.".format(1000 * (time.time() - start_time)))

    start_time = time.time()
    head_pose_estimation_model = HeadPoseEstimationModel(head_pose_estimation_model_file, device_name, cpu_extension)
    logging.info("Head Pose Estimation Model: {:.1f} ms.".format(1000 * (time.time() - start_time)))

    start_time = time.time()
    gaze_estimation_model = GazeEstimationModel(gaze_estimation_model_file, device_name, cpu_extension)
    logging.info("Gaze Estimation Model: {:.1f} ms.".format(1000 * (time.time() - start_time)))

    logging.info("*********** Model Load Completed ***********")

    feeder = InputFeeder('video', video_file)
    feeder.load_data()

    frame_count = 0
    face_detect_infer_time = 0

    while True:
        try:
            frame = next(feeder.next_batch())
        except StopIteration:
            break

        key_pressed = cv2.waitKey(60)
        frame_count += 1

        image = face_detection_model.preprocess_input(frame)

        start_time = time.time()
        outputs = face_detection_model.predict(image)
        face_detect_infer_time += (time.time() - start_time)
        out_frame, faces = face_detection_model.preprocess_output(outputs, frame, prob_threshold)

        if key_pressed == 27:
            break

    if frame_count > 0:
        logging.info("*********** Model Inference Time ****************")
        logging.info("Face Detection Model: {:.1f} ms.".format(1000 * face_detect_infer_time / frame_count))
        logging.info("*********** Model Inference Completed ***********")

    feeder.close()
    cv2.destroyAllWindows()

def main():
    args = build_argparser().parse_args()
    logfile_config()
    infer_on_stream(args)


if __name__ == '__main__':
    main()