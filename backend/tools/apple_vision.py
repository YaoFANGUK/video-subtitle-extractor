""" Use Apple's Vision Framework via PyObjC to detect text in images """

import pathlib

import Quartz
import Vision
from Cocoa import NSURL
from Foundation import NSDictionary
# needed to capture system-level stderr
from wurlitzer import pipes
import time
from pprint import pprint
import cv2
from tools.pyci import cimg

class OcrRecogniser:
    def __init__(self):
        self.results = []
        self.handler = make_request_handler(self.results)
        self.vision_request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(self.handler)
        self.vision_request.setRecognitionLanguages_(['zh-CN', 'en'])
        #self.vision_request.setRecognitionLevel_(0)
        self.vision = Vision.VNImageRequestHandler.alloc()

    def predict(self, image):
        # a = time.time()
        self.height, self.width = image.shape[0:2]
        # img_path = "/Users/jason/tmp/1.raw"
        # cv2.imwrite(img_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #
        # b = time.time()
        # print(b-a)
        # input_url = NSURL.fileURLWithPath_(img_path)
        # with pipes() as (out, err):
        #     # capture stdout and stderr from system calls
        #     # otherwise, Quartz.CIImage.imageWithContentsOfURL_
        #     # prints to stderr something like:
        #     # 2020-09-20 20:55:25.538 python[73042:5650492] Creating client/daemon connection: B8FE995E-3F27-47F4-9FA8-559C615FD774
        #     # 2020-09-20 20:55:25.652 python[73042:5650492] Got the query meta data reply for: com.apple.MobileAsset.RawCamera.Camera, response: 0
        #     input_image = Quartz.CIImage.imageWithContentsOfURL_(input_url)
        input_image = cimg(image).ciimage
        vision_options = NSDictionary.dictionaryWithDictionary_({})
        vision_handler = self.vision.initWithCIImage_options_(
            input_image, vision_options
        )

        error = vision_handler.performRequests_error_([self.vision_request], None)
        # vision_handler.dealloc()
        return [], self.results

    def get_coordinates(self, dt_box):
        coordinate_list = list()
        for result in  self.results:
            xmin, ymin, xmax, ymax = result[2][0:4]
            xmin = xmin * self.width
            xmax = xmax * self.width
            ymin = self.height - ymin * self.height
            ymax = self.height - ymax * self.height
            coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

def make_request_handler(results):
    """ results: list to store results """
    if not isinstance(results, list):
        raise ValueError("results must be a list")

    def handler(request, error):
        if error:
            print(f"Error! {error}")
        else:
            observations = request.results()
            results.clear()
            for text_observation in observations:
                recognized_text = text_observation.topCandidates_(1)[0]
                xmin = text_observation.topLeft().x
                ymin = text_observation.topLeft().y
                xmax = text_observation.bottomRight().x
                ymax = text_observation.bottomRight().y
                results.append([recognized_text.string(), recognized_text.confidence(), (xmin, ymin, xmax, ymax)])
    return handler