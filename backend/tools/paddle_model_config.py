import os
import re
from pathlib import Path
from fsplit.filesplit import Filesplit
from backend.config import BASE_DIR, config

class PaddleModelConfig:
    def __init__(self, hardware_accelerator):
        self.hardware_accelerator = hardware_accelerator
        # 设置识别语言
        self.REC_CHAR_TYPE = config.language.value

        # 模型文件目录
        # 默认模型版本 V4
        self.MODEL_VERSION = 'V4'
        # 文本检测模型
        self.DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
        # 设置文本识别模型 + 字典
        self.REC_MODEL_BASE = os.path.join(BASE_DIR, 'models')
        # V3, V4模型默认图形识别的shape为3, 48, 320
        self.REC_IMAGE_SHAPE = '3,48,320'
        self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'{self.REC_CHAR_TYPE}_rec')
        self.DET_MODEL_PATH = os.path.join(self.DET_MODEL_BASE, self.MODEL_VERSION, f'{self.REC_CHAR_TYPE}_det')

        self.LATIN_LANG = [
            'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
            'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
            'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
            'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french'
        ]
        self.ARABIC_LANG = ['ar', 'fa', 'ug', 'ur']
        self.CYRILLIC_LANG = [
            'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
            'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic'
        ]
        self.DEVANAGARI_LANG = [
            'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
            'sa', 'bgc', 'devanagari'
        ]
        self.OTHER_LANG = [
            'ch', 'japan', 'korean', 'en', 'ta', 'kn', 'te', 'ka',
            'chinese_cht',
        ]
        self.MULTI_LANG = self.LATIN_LANG + self.ARABIC_LANG + self.CYRILLIC_LANG + self.DEVANAGARI_LANG + \
                    self.OTHER_LANG

        self.DET_MODEL_FAST_PATH = os.path.join(self.DET_MODEL_BASE, self.MODEL_VERSION, 'ch_det_fast')

        # 如果设置了识别文本语言类型，则设置为对应的语言
        if self.REC_CHAR_TYPE in self.MULTI_LANG:
            # 定义文本检测与识别模型
            # 使用快速模式时，调用轻量级模型
            if config.mode.value == 'fast':
                self.DET_MODEL_PATH = os.path.join(self.DET_MODEL_BASE, self.MODEL_VERSION, 'ch_det_fast')
                self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'{self.REC_CHAR_TYPE}_rec_fast')
            # 使用自动模式时，检测有没有使用GPU，根据GPU判断模型
            elif config.mode.value == 'auto':
                # 如果使用GPU，则使用大模型
                if self.hardware_accelerator.has_accelerator():
                    self.DET_MODEL_PATH = os.path.join(self.DET_MODEL_BASE, self.MODEL_VERSION, 'ch_det')
                    # 英文模式的ch模型识别效果好于fast
                    if self.REC_CHAR_TYPE == 'en':
                        self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'ch_rec')
                    else:
                        self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'{self.REC_CHAR_TYPE}_rec')
                else:
                    self.DET_MODEL_PATH = os.path.join(self.DET_MODEL_BASE, self.MODEL_VERSION, 'ch_det_fast')
                    self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'{self.REC_CHAR_TYPE}_rec_fast')
            else:
                self.DET_MODEL_PATH = os.path.join(self.DET_MODEL_BASE, self.MODEL_VERSION, 'ch_det')
                self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'{self.REC_CHAR_TYPE}_rec')
            # 如果默认版本(V4)没有大模型，则切换为默认版本(V4)的fast模型
            if not os.path.exists(self.REC_MODEL_PATH):
                self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'{self.REC_CHAR_TYPE}_rec_fast')
            # 如果默认版本(V4)既没有大模型，又没有fast模型，则使用V3版本的大模型
            if not os.path.exists(self.REC_MODEL_PATH):
                self.MODEL_VERSION = 'V3'
                self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'{self.REC_CHAR_TYPE}_rec')
            # 如果V3版本没有大模型，则使用V3版本的fast模型
            if not os.path.exists(self.REC_MODEL_PATH):
                self.MODEL_VERSION = 'V3'
                self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'{self.REC_CHAR_TYPE}_rec_fast')

            if self.REC_CHAR_TYPE in self.LATIN_LANG:
                self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'latin_rec_fast')
            elif self.REC_CHAR_TYPE in self.ARABIC_LANG:
                self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'arabic_rec_fast')
            elif self.REC_CHAR_TYPE in self.CYRILLIC_LANG:
                self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'cyrillic_rec_fast')
            elif self.REC_CHAR_TYPE in self.DEVANAGARI_LANG:
                self.REC_MODEL_PATH = os.path.join(self.REC_MODEL_BASE, self.MODEL_VERSION, f'devanagari_rec_fast')

            # 定义图像识别shape
            if self.MODEL_VERSION == 'V2':
                self.REC_IMAGE_SHAPE = '3,32,320'
            else:
                self.REC_IMAGE_SHAPE = '3,48,320'

            # 查看该路径下是否有文本模型识别完整文件，没有的话合并小文件生成完整文件
            if 'inference.pdiparams' not in (os.listdir(self.REC_MODEL_PATH)):
                fs = Filesplit()
                fs.merge(input_dir=self.REC_MODEL_PATH)
            # 查看该路径下是否有文本模型识别完整文件，没有的话合并小文件生成完整文件
            if 'inference.pdiparams' not in (os.listdir(self.DET_MODEL_PATH)):
                fs = Filesplit()
                fs.merge(input_dir=self.DET_MODEL_PATH)
   
    def convertToOnnxModelIfNeeded(self, model_dir, model_filename="inference.pdmodel", params_filename="inference.pdiparams", opset_version=14):
        """Converts a Paddle model to ONNX if ONNX providers are available and the model does not already exist."""
        
        if not self.hardware_accelerator.onnx_providers:
            return model_dir
        
        onnx_model_path = os.path.join(model_dir, "model.onnx")

        if os.path.exists(onnx_model_path):
            print(f"ONNX model already exists: {onnx_model_path}. Skipping conversion.")
            return onnx_model_path
        
        print(f"Converting Paddle model {model_dir} to ONNX...")
        model_file = os.path.join(model_dir, model_filename)
        params_file = os.path.join(model_dir, params_filename) if params_filename else ""

        try:
            import paddle2onnx
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

            # Convert and save the model
            onnx_model = paddle2onnx.export(
                model_filename=model_file,
                params_filename=params_file,
                save_file=onnx_model_path,
                opset_version=opset_version,
                auto_upgrade_opset=True,
                verbose=True,
                enable_onnx_checker=True,
                enable_experimental_op=True,
                enable_optimize=True,
                custom_op_info={},
                deploy_backend="onnxruntime",
                calibration_file="calibration.cache",
                external_file=os.path.join(model_dir, "external_data"),
                export_fp16_model=False,
            )

            print(f"Conversion successful. ONNX model saved to: {onnx_model_path}")
            return onnx_model_path
        except Exception as e:
            print(f"Error during conversion: {e}")
            return model_dir