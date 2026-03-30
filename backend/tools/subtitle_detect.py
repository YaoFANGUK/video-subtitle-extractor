
from .paddle_model_config import PaddleModelConfig
from .hardware_accelerator import HardwareAccelerator

class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self):
        from paddleocr.tools.infer import utility
        from paddleocr.tools.infer.predict_det import TextDetector
        hardware_accelerator = HardwareAccelerator.instance()
        onnx_providers = hardware_accelerator.onnx_providers
        model_config = PaddleModelConfig(hardware_accelerator)
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = model_config.convertToOnnxModelIfNeeded(model_config.DET_MODEL_PATH)
        args.use_gpu=hardware_accelerator.has_cuda()
        args.use_onnx=len(onnx_providers) > 0
        args.onnx_providers=onnx_providers
        self.text_detector = TextDetector(args)

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse
