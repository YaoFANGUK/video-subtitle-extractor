import os
import traceback
from backend.config import *
import importlib
from paddleocr import PaddleOCR
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.paddle_model_config import PaddleModelConfig

# PyInstaller compatibility: bypass paddlex dependency checks
# importlib.metadata can't find dist-info dirs in the bundle, causing false alarms.
# All deps are bundled, so the checks are safe to skip.
try:
    import paddlex.utils.deps as _pdeps
    import importlib.util as _il_util

    # Patch is_dep_available: use find_spec for special deps, True for everything else
    def _patched_is_dep_available(dep, /, check_version=False):
        _special = {"paddlepaddle": "paddle", "paddle-custom-device": "paddle_custom_device",
                    "ultra-infer": "ultra_infer", "fastdeploy": "fastdeploy",
                    "onnxruntime": "onnxruntime"}
        if dep in _special:
            return _il_util.find_spec(_special[dep]) is not None
        return True

    _pdeps.is_dep_available = _patched_is_dep_available
    # Safety net: also bypass require_extra and require_deps
    _pdeps.require_extra = lambda *a, **kw: None
    _pdeps.require_deps = lambda *a, **kw: None
except Exception as e:
    print(f"Warning: failed to patch paddlex deps: {e}")

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

        # PaddleOCR 3.x: 使用 predict_iter 获取结果
        results = list(self.recogniser.predict_iter(image))
        if not results:
            return [], []

        res = results[0]
        dt_polys = res.get('dt_polys', [])
        rec_texts = res.get('rec_texts', [])
        rec_scores = res.get('rec_scores', [])

        if len(dt_polys) == 0:
            return [], []

        # 将 dt_polys (numpy array, shape (N, points, 2)) 转换为旧的 dt_box 格式
        # 旧格式: [[(x1,y1),(x2,y2),(x3,y3),(x4,y4)], ...]
        dt_box = []
        coordinate_list = []
        for poly in dt_polys:
            points = [(int(p[0]), int(p[1])) for p in poly]
            # 取 AABB 用于排序
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            coordinate_list.append([xmin, xmax, ymin, ymax])
            dt_box.append(points)

        # 将 rec_texts + rec_scores 转换为旧的 rec_res 格式
        rec_res = [(text, float(score)) for text, score in zip(rec_texts, rec_scores)]

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

        to_rank_res = list(zip(coordinate_list, rec_res, dt_box))
        # 用sorted替代冒泡排序：先按ymin，再按xmin
        ranked_res = sorted(to_rank_res, key=lambda x: (x[0][2], x[0][0]))
        # 重建 dt_box 和 rec_res（排序后）
        sorted_dt_box = []
        sorted_rec_res = []
        for coord, rec, box in ranked_res:
            # 将 coordinate 转换回 4 点格式
            xmin, xmax, ymin, ymax = coord
            sorted_dt_box.append([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            sorted_rec_res.append(rec)

        return sorted_dt_box, sorted_rec_res

    def init_model(self):
        model_config = PaddleModelConfig(self.hardware_accelerator)

        # PaddleOCR 3.x 使用 device 参数替代 use_gpu
        if self.hardware_accelerator.has_cuda():
            device = 'gpu:0'
        else:
            device = 'cpu'

        kwargs = dict(
            text_detection_model_dir=model_config.DET_MODEL_PATH,
            text_recognition_model_dir=model_config.REC_MODEL_PATH,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=0,
            device=device,
        )
        if model_config.DET_MODEL_NAME:
            kwargs['text_detection_model_name'] = model_config.DET_MODEL_NAME
        if model_config.REC_MODEL_NAME:
            kwargs['text_recognition_model_name'] = model_config.REC_MODEL_NAME

        try:
            return PaddleOCR(**kwargs)
        except Exception as e:
            # Print full error chain for debugging
            print(f"Error initializing PaddleOCR: {e}")
            if e.__cause__:
                print(f"  Caused by: {e.__cause__}")
            traceback.print_exc()
            raise


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
