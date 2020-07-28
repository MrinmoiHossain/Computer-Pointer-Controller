import os
import cv2
import logging
from openvino.inference_engine import IECore, IENetwork

class Model:
    def __init__(self):
        self.core = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device = 'CPU', cpu_extensions = None):
        model_structure = model
        model_weights = os.path.splitext(model_structure)[0] + ".bin"

        self.core = IECore()
        self.network = IENetwork(model = model_structure, weights = model_weights)

        if cpu_extensions is not None:
            self.core.add_extension(cpu_extension, device)

        supported_layers = self.core.query_network(network = self.network, device_name = device)
        
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) != 0:
            logging.error("Unsupported layers found: {}".format(unsupported_layers))
            logging.error("Check whether extensions are available to add to IECore.")
            exit(1)
            
        self.exec_network = self.core.load_network(self.network, device_name = device, num_requests = 1)

        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

    def get_input_shape(self):
        return self.network.inputs[self.input_blob].shape

    def preprocess_input(self, image):
        input_shape = self.get_input_shape()

        image = cv2.resize(image, (input_shape[3], input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        return image

    def exec_net(self, image, request_id = 0):
        return self.exec_network.start_async(request_id, inputs = {self.input_blob: image})

    def wait(self):
        return self.exec_network.requests[0].wait(-1)