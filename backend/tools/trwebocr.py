import time

import cv2
import io
import requests
import json
import numpy as np
import urllib
import traceback

# https://github.com/alisen39/TrWebOCR/wiki/%E6%8E%A5%E5%8F%A3%E6%96%87%E6%A1%A3

DEBUG_OCR_OUTPUT = False
DEBUG_OCR_OUTPUT_LOCATION = 'R:/TEMP/vse'
class OcrRecogniser:
    def __init__(self):
        self.ocr_fullscreen_mode = False

    def predict(self, id, image, sub_area):
        try:
            self.height, self.width = image.shape[0:2]
            font_size = int(min(self.height, self.width) / 5)
            sub_area = np.array(sub_area)
            if self.ocr_fullscreen_mode:
                sub_area_fix = sub_area
            else:
                sub_area_fix = sub_area + (-font_size, +font_size, -font_size, +font_size)
            f_ymin, f_ymax, f_xmin, f_xmax = sub_area_fix
            f_ymin = int(max(f_ymin, 0))
            f_xmin = int(max(f_xmin, 0))
            f_ymax = int(min(f_ymax, self.height))
            f_xmax = int(min(f_xmax, self.width))
            # print(f"trwebocr image ymin:{f_ymin}, ymax:{f_ymax}, xmin:{f_xmin}, xmax:{f_xmax}")
            is_success, buffer = cv2.imencode(".jpg", image[f_ymin:f_ymax, f_xmin:f_xmax])
            url = 'http://127.0.0.1:18089/api/tr-run/'
            res = requests.post(url=url, data={'compress': 0, 'is_draw': 1 if DEBUG_OCR_OUTPUT else 0},
                                files={'file': io.BytesIO(buffer)}, timeout=5)

            if not res.ok:
                print(f"trwebocr image ymin:{f_ymin}, ymax:{f_ymax}, xmin:{f_xmin}, xmax:{f_xmax}")
                print("trwebocr res ", res.text)
                return [], []
            # result = res.content.decode('unicode-escape')
            # print("trwebocr", res.text)
            res = json.loads(res.text)
            if DEBUG_OCR_OUTPUT:
                response = urllib.request.urlopen(res['data']['img_detected'])
                with open(f'{DEBUG_OCR_OUTPUT_LOCATION}/{id}.jpg', 'wb') as f:
                    f.write(response.file.read())
            # print(res['data']['raw_out'])
            rec_res = list()
            dt_box = list()
            for it in res['data']['raw_out']:
                rec_res.append([it[1], it[2]])
                x, y, width, height, angle = it[0]
                x = x + f_xmin
                y = y + f_ymin
                width_half = width / 2
                height_half = height / 2
                # tr的框大了点, 适当缩小
                adjust_size = height_half / 5
                xmin = max(int(x - width_half - 0.5 + adjust_size), f_xmin)
                ymin = max(int(y - height_half - 0.5 + adjust_size), f_ymin)
                xmax = min(int(x + width_half - 0.5 - adjust_size), f_xmax)
                ymax = min(int(y + height_half - 0.5 - adjust_size), f_ymax)
                dt_box.append((xmin, xmax, ymin, ymax))
            return dt_box, rec_res
        except requests.exceptions.ConnectionError:
            traceback.print_exc()
            return [], []

    def get_coordinates(self, dt_box):
        return dt_box