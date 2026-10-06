
import backend.tools.quiet_paddle  # noqa: F401  must precede paddleocr
from backend.tools.paddle_model_config import PaddleModelConfig
from backend.tools.hardware_accelerator import HardwareAccelerator
import numpy as np

try:
    from paddleocr import TextDetection
except ImportError:
    TextDetection = None


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self):
        hardware_accelerator = HardwareAccelerator.instance()
        model_config = PaddleModelConfig(hardware_accelerator)
        # 使用 TextDetection 公开 API（PaddleOCR 3.x）
        kwargs = {'model_dir': model_config.DET_MODEL_PATH}
        if model_config.DET_MODEL_NAME:
            kwargs['model_name'] = model_config.DET_MODEL_NAME
        self.text_detector = TextDetection(**kwargs)

    def detect_subtitle(self, img, roi=None):
        """
        检测图像中的文本框
        :param img: 输入图像
        :param roi: optional region of interest; returned boxes remain in frame coordinates
        :return: (dt_boxes, elapse) dt_boxes为numpy数组，elapse为耗时
        """
        model_img = roi.crop(img) if roi is not None else img
        results = list(self.text_detector.predict(model_img))
        if not results:
            return np.array([]), 0
        res = results[0]
        dt_polys = res.get('dt_polys', np.array([]))
        if roi is not None and len(dt_polys):
            polygons = np.asarray(dt_polys)
            dt_polys = np.asarray([
                roi.to_original(poly.tolist()) for poly in polygons
            ], dtype=polygons.dtype)
        return dt_polys, 0
