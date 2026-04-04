import os
from backend.config import *
import importlib
from paddleocr import PaddleOCR
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.paddle_model_config import PaddleModelConfig

# 加载文本检测+识别模型
class OcrRecogniser:
    def __init__(self):
        self.recogniser = None
        # 占位，应该由main.py初始化
        self.hardware_accelerator = HardwareAccelerator()

    @staticmethod
    def y_round(y):
        y_min = y + 10 - y % 10
        y_max = y - y % 10
        if abs(y - y_min) < abs(y - y_max):
            return y_min
        else:
            return y_max

    def predict(self, image):
        if not self.recogniser:
            self.recogniser = self.init_model()
        detection_box, recognise_result, _ = self.recogniser(image, cls=False)
        if len(detection_box) > 0:
            coordinate_list = list()
            if isinstance(detection_box, list):
                for i in detection_box:
                    i = list(i)
                    (x1, y1) = int(i[0][0]), int(i[0][1])
                    (x2, y2) = int(i[1][0]), int(i[1][1])
                    (x3, y3) = int(i[2][0]), int(i[2][1])
                    (x4, y4) = int(i[3][0]), int(i[3][1])
                    xmin = max(x1, x4)
                    xmax = min(x2, x3)
                    ymin = max(y1, y2)
                    ymax = min(y3, y4)
                    coordinate_list.append([xmin, xmax, ymin, ymax])

            # 计算有多少行字幕，将每行字幕最小的ymin值放入lines
            lines = []
            for i in coordinate_list:
                rounded_y = self.y_round(i[2])
                if not any(abs(rounded_y - line_y) <= 10 for line_y in lines):
                    lines.append(rounded_y)
            lines = sorted(lines)

            for i in coordinate_list:
                for j in lines:
                    if abs(j - self.y_round(i[2])) <= 10:
                        i[2] = j

            to_rank_res = list(zip(coordinate_list, recognise_result))
            # 用sorted替代冒泡排序：先按ymin，再按xmin
            ranked_res = sorted(to_rank_res, key=lambda x: (x[0][2], x[0][0]))
            dt_box = []
            for i in [j[0] for j in ranked_res]:
                dt_box.append([(i[0], i[2]), (i[1], i[2]), (i[1], i[3]), (i[0], i[3])])
            res = [i[1] for i in ranked_res]
            return dt_box, res
        else:
            return detection_box, recognise_result

    def init_model(self):
        model_config = PaddleModelConfig(self.hardware_accelerator)
        onnx_providers = self.hardware_accelerator.onnx_providers
        return PaddleOCR(
            use_gpu=self.hardware_accelerator.has_cuda(),
            gpu_mem=500,
            det_algorithm='DB',
            # 设置文本检测模型路径
            det_model_dir=model_config.convertToOnnxModelIfNeeded(model_config.DET_MODEL_PATH),
            rec_algorithm='CRNN',
            # 设置每张图文本框批处理数量
            rec_batch_num=config.recBatchNumber.value,
            # 设置文本识别模型路径
            rec_model_dir=model_config.convertToOnnxModelIfNeeded(model_config.REC_MODEL_PATH),
            max_batch_size=config.maxBatchSize.value,
            det=True,
            use_angle_cls=False,
            drop_score=0,
            lang=model_config.REC_CHAR_TYPE,
            ocr_version=f'PP-OCR{model_config.MODEL_VERSION.lower()}',
            rec_image_shape=model_config.REC_IMAGE_SHAPE,
            use_onnx=len(onnx_providers) > 0,
            onnx_providers=onnx_providers,
            debug=False, 
            show_log=False,
        )
    
def get_coordinates(dt_box):
    """
    从返回的检测框中获取坐标
    :param dt_box 检测框返回结果
    :return list 坐标点列表
    """
    coordinate_list = list()
    if isinstance(dt_box, list):
        for i in dt_box:
            i = list(i)
            (x1, y1) = int(i[0][0]), int(i[0][1])
            (x2, y2) = int(i[1][0]), int(i[1][1])
            (x3, y3) = int(i[2][0]), int(i[2][1])
            (x4, y4) = int(i[3][0]), int(i[3][1])
            xmin = max(x1, x4)
            xmax = min(x2, x3)
            ymin = max(y1, y2)
            ymax = min(y3, y4)
            coordinate_list.append((xmin, xmax, ymin, ymax))
    return coordinate_list
