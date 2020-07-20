import os
import time
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import FaceDetectionModel

def build_argparser():
    parser = ArgumentParser()

    parser.add_argument("-mfd", "--faceDetectionModel", type=str, required=True,
                        help="Specify path of xml file of face detection model")

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
    

def infer_on_stream(args):
    face_detection_model_file = args.mfd

    input_file = args.i
    device_name = args.d
    cpu_extension = args.l
    prob_threshold = args.pt

    start_load_time = time.time()
    face_detection_model = FaceDetectionModel(face_detection_model_file, device_name, cpu_extension)

    total_load_time = time.time() - start_load_time

    feeder = InputFeeder('video', input_file)
    feeder.load_data()
    ## TODO

    ## TODO
    feeder.close()

def main():
    agrs = build_argparser().parse_args()
    infer_on_stream(args)


if __name__ == '__main__':
    main()