import os
import sys
import time
import cv2
import logging
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from gaze_estimation import GazeEstimationModel
from mouse_controller import MouseController

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

    parser.add_argument("-o", "--output_path", default='result/', type=str,
                        help="Output video path")

    parser.add_argument("-sv", "--show_video", type=str, default='no',
                        help="Output video show mode")                    

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
    show_video = args.show_video

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    mouse_control = MouseController("low", "fast")

    try:
        logging.info("*********** Model Load Time ***************")
        start_model_load_time = time.time()

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

        total_model_load_time = time.time() - start_model_load_time
        logging.info("*********** Model Load Completed ***********")
    except Exception as e:
        logging.error("ERROR in model loading: " + str(e))
        sys.exit(1)


    feeder = InputFeeder('video', video_file)
    feeder.load_data()

    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), int(feeder.fps()/10), (1920, 1080), True)

    start_inference_time = 0
    frame_count = 0
    face_detect_infer_time = 0
    facial_landmarks_infer_time = 0
    head_pose_infer_time = 0
    gaze_infer_time = 0

    while True:
        try:
            frame = next(feeder.next_batch())
        except StopIteration:
            break

        key_pressed = cv2.waitKey(60)
        frame_count += 1

        ## Face Detecton Model
        image = face_detection_model.preprocess_input(frame)

        start_time = time.time()
        outputs = face_detection_model.predict(image)
        face_detect_infer_time += (time.time() - start_time)
        out_frame, faces = face_detection_model.preprocess_output(outputs, frame, prob_threshold)

        for face in faces:
            crop_image = frame[face[1]:face[3], face[0]:face[2]]

            ## Facial Landmarks Detecton Model
            image = facial_landmarks_detection_model.preprocess_input(crop_image)

            start_time = time.time()
            outputs = facial_landmarks_detection_model.predict(image)
            facial_landmarks_infer_time += (time.time() - start_time)
            out_frame, left_eye_point, right_eye_point = facial_landmarks_detection_model.preprocess_output(outputs, out_frame, face)

            ## Head Pose Estimation Model
            image = head_pose_estimation_model.preprocess_input(crop_image)

            start_time = time.time()
            outputs = head_pose_estimation_model.predict(image)
            head_pose_infer_time += (time.time() - start_time)
            out_frame, headpose_angels_list = head_pose_estimation_model.preprocess_output(outputs, out_frame)

            ## Gaze Estimation Model
            out_frame, left_eye, right_eye  = gaze_estimation_model.preprocess_input(out_frame, crop_image, left_eye_point, right_eye_point)

            start_time = time.time()
            outputs = gaze_estimation_model.predict(left_eye, right_eye, headpose_angels_list)
            gaze_infer_time += (time.time() - start_time)
            out_frame, gazevector = gaze_estimation_model.preprocess_output(outputs, out_frame, face, left_eye_point, right_eye_point)

            if show_video == 'yes':
                cv2.imshow("Computer Pointer Control", out_frame)
                out_video.write(out_frame)
                mouse_control.move(gazevector[0], gazevector[1])

        if key_pressed == 27:
            break

    if frame_count > 0:
        logging.info("*********** Model Inference Time ****************")
        logging.info("Face Detection Model: {:.1f} ms.".format(1000 * face_detect_infer_time / frame_count))
        logging.info("Facial Landmarks Detection Model: {:.1f} ms.".format(1000 * facial_landmarks_infer_time / frame_count))
        logging.info("Head Pose Detection Model: {:.1f} ms.".format(1000 * head_pose_infer_time / frame_count))
        logging.info("Gaze Detection Model: {:.1f} ms.".format(1000 * gaze_infer_time / frame_count))
        logging.info("*********** Model Inference Completed ***********")

    total_infer_time = time.time() - start_inference_time
    total_inference_time = round(total_infer_time, 1)
    fps = frame_count / total_inference_time
    
    with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
        f.write(str(total_inference_time)+'\n')
        f.write(str(fps)+'\n')
        f.write(str(total_model_load_time)+'\n')

    logging.info("*********** Total Summary ****************")
    logging.info(f"Total Model Load Time: {total_model_load_time}")
    logging.info(f"Total Inference Time: {total_inference_time}")
    logging.info(f"FPS: {fps}")
    logging.info("*********** Total Summary ***********")
    logging.info("*********** ************************* ***********")

    feeder.close()
    cv2.destroyAllWindows()

def main():
    args = build_argparser().parse_args()
    logfile_config()
    infer_on_stream(args)


if __name__ == '__main__':
    main()