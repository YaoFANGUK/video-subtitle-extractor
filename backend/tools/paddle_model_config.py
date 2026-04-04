import os
from pathlib import Path
from fsplit.filesplit import Filesplit
from backend.config import BASE_DIR, config


class PaddleModelConfig:
    def __init__(self, hardware_accelerator):
        self.hardware_accelerator = hardware_accelerator
        # 设置识别语言
        self.REC_CHAR_TYPE = config.language.value

        # 模型文件目录
        self.MODEL_BASE = os.path.join(BASE_DIR, 'models')
        # 默认模型版本 V5
        self.MODEL_VERSION = 'V5'
        # V3, V4, V5模型默认图形识别的shape为3, 48, 320
        self.REC_IMAGE_SHAPE = '3,48,320'
        # 初始化模型路径
        self.REC_MODEL_PATH = None
        self.DET_MODEL_PATH = None
        self.DET_MODEL_NAME = None
        self.REC_MODEL_NAME = None

        # 语言组定义
        self.LATIN_LANG = [
            'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
            'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
            'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
            'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french',
            'fi', 'eu', 'gl', 'lb', 'rm', 'ca', 'qu',
        ]
        self.ARABIC_LANG = ['ar', 'fa', 'ug', 'ur', 'ps', 'sd', 'bal']
        self.CYRILLIC_LANG = [
            'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
            'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic',
            'sr', 'kk', 'ky', 'tg', 'mk', 'tt', 'cv', 'ba', 'mhr', 'mo',
            'udm', 'kv', 'os', 'bua', 'xal', 'tyv', 'sah', 'kaa',
        ]
        self.ESLAV_LANG = ['ru', 'be', 'uk']
        self.DEVANAGARI_LANG = [
            'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
            'sa', 'bgc', 'devanagari',
        ]
        self.OTHER_LANG = [
            'ch', 'japan', 'korean', 'en', 'ta', 'kn', 'te', 'ka',
            'chinese_cht',
        ]
        self.MULTI_LANG = (self.LATIN_LANG + self.ARABIC_LANG + self.CYRILLIC_LANG
                           + self.DEVANAGARI_LANG + self.OTHER_LANG)

        # 如果设置了识别文本语言类型，则设置为对应的语言
        if self.REC_CHAR_TYPE in self.MULTI_LANG:
            # 优先尝试 V5 模型
            v5_resolved = self._try_resolve_v5_models()
            if v5_resolved:
                self.MODEL_VERSION = 'V5'
                self.DET_MODEL_PATH, self.REC_MODEL_PATH, self.DET_MODEL_NAME, self.REC_MODEL_NAME = v5_resolved
            else:
                # Fallback 到 V4/V3
                self._resolve_legacy_models()

            # 定义图像识别shape
            if self.MODEL_VERSION == 'V2':
                self.REC_IMAGE_SHAPE = '3,32,320'
            else:
                self.REC_IMAGE_SHAPE = '3,48,320'

            # 查看该路径下是否有文本模型识别完整文件，没有的话合并小文件生成完整文件
            if self.REC_MODEL_PATH and 'inference.pdiparams' not in os.listdir(self.REC_MODEL_PATH):
                fs = Filesplit()
                fs.merge(input_dir=self.REC_MODEL_PATH)
            if self.DET_MODEL_PATH and 'inference.pdiparams' not in os.listdir(self.DET_MODEL_PATH):
                fs = Filesplit()
                fs.merge(input_dir=self.DET_MODEL_PATH)

    def _get_v5_rec_model_name(self, lang):
        """
        根据语言获取V5识别模型目录名
        参考: https://www.paddleocr.ai/main/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
        """
        if lang in ('ch', 'chinese_cht', 'japan'):
            return 'PP-OCRv5_server_rec_infer'
        elif lang == 'en':
            # en 可以用 server 模型(中英日通用) 也可以用专用 mobile 模型
            # 这里优先使用 server 模型，精度更高
            return 'PP-OCRv5_server_rec_infer'
        elif lang == 'korean':
            return 'korean_PP-OCRv5_mobile_rec_infer'
        elif lang in self.LATIN_LANG:
            return 'latin_PP-OCRv5_mobile_rec_infer'
        elif lang in self.ARABIC_LANG:
            return 'arabic_PP-OCRv5_mobile_rec_infer'
        elif lang in self.CYRILLIC_LANG:
            return 'cyrillic_PP-OCRv5_mobile_rec_infer'
        elif lang in self.DEVANAGARI_LANG:
            return 'devanagari_PP-OCRv5_mobile_rec_infer'
        elif lang == 'th':
            return 'th_PP-OCRv5_mobile_rec_infer'
        elif lang == 'el':
            return 'el_PP-OCRv5_mobile_rec_infer'
        elif lang == 'ta':
            return 'ta_PP-OCRv5_mobile_rec_infer'
        elif lang == 'te':
            return 'te_PP-OCRv5_mobile_rec_infer'
        return None

    @staticmethod
    def _read_model_name_from_yaml(model_dir):
        """从 inference.yml 中读取 Global.model_name"""
        yaml_path = os.path.join(model_dir, 'inference.yml')
        if not os.path.exists(yaml_path):
            return None
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                in_global = False
                for line in f:
                    stripped = line.strip()
                    if stripped == 'Global:':
                        in_global = True
                        continue
                    if in_global:
                        if stripped and not stripped.startswith('#') and ':' in stripped:
                            if stripped.startswith('model_name:'):
                                return stripped.split(':', 1)[1].strip().strip('"').strip("'")
                        # 遇到下一个顶级 section 则退出
                        if stripped and not stripped.startswith('model_name') and not stripped.startswith(' ') and stripped.endswith(':'):
                            break
        except Exception:
            pass
        return None

    def _try_resolve_v5_models(self):
        """
        尝试解析 V5 模型路径，返回 (det_model_path, rec_model_path, det_model_name, rec_model_name) 或 None
        """
        v5_base = os.path.join(self.MODEL_BASE, 'V5')

        # 快速模式优先使用 mobile 模型，否则使用 server 模型
        if config.mode.value == 'fast':
            det_model_path = os.path.join(v5_base, 'PP-OCRv5_mobile_det_infer')
            if not os.path.exists(det_model_path):
                det_model_path = os.path.join(v5_base, 'PP-OCRv5_server_det_infer')
        else:
            det_model_path = os.path.join(v5_base, 'PP-OCRv5_server_det_infer')
        if not os.path.exists(det_model_path):
            return None

        det_model_name = self._read_model_name_from_yaml(det_model_path)

        # 快速模式使用 mobile 识别模型
        if config.mode.value == 'fast':
            rec_model_path = os.path.join(v5_base, 'PP-OCRv5_mobile_rec_infer')
            if os.path.exists(rec_model_path):
                rec_model_name = self._read_model_name_from_yaml(rec_model_path)
                return det_model_path, rec_model_path, det_model_name, rec_model_name
            # mobile 不存在则 fallback 到按语言选择

        # 获取识别模型
        rec_model_dir_name = self._get_v5_rec_model_name(self.REC_CHAR_TYPE)
        if rec_model_dir_name is None:
            return None

        rec_model_path = os.path.join(v5_base, f'{rec_model_dir_name}_infer'
                                      if not rec_model_dir_name.endswith('_infer')
                                      else rec_model_dir_name)

        if not os.path.exists(rec_model_path):
            rec_model_path = os.path.join(v5_base, rec_model_dir_name)

        if not os.path.exists(rec_model_path):
            return None

        rec_model_name = self._read_model_name_from_yaml(rec_model_path)
        return det_model_path, rec_model_path, det_model_name, rec_model_name

    def _resolve_legacy_models(self):
        """
        Fallback: 使用 V4/V3 模型
        """
        # 尝试 V4
        self.MODEL_VERSION = 'V4'

        if config.mode.value == 'fast':
            self.DET_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION, 'ch_det_fast')
            self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION,
                                               f'{self.REC_CHAR_TYPE}_rec_fast')
        elif config.mode.value == 'auto':
            if self.hardware_accelerator.has_accelerator():
                self.DET_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION, 'ch_det')
                if self.REC_CHAR_TYPE == 'en':
                    self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION, f'ch_rec')
                else:
                    self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION,
                                                       f'{self.REC_CHAR_TYPE}_rec')
            else:
                self.DET_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION, 'ch_det_fast')
                self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION,
                                                   f'{self.REC_CHAR_TYPE}_rec_fast')
        else:
            self.DET_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION, 'ch_det')
            self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION,
                                               f'{self.REC_CHAR_TYPE}_rec')

        # 如果V4没有大模型，则切换为fast模型
        if not os.path.exists(self.REC_MODEL_PATH):
            self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION,
                                               f'{self.REC_CHAR_TYPE}_rec_fast')
        # 如果V4既没有大模型又没有fast模型，则使用V3
        if not os.path.exists(self.REC_MODEL_PATH):
            self.MODEL_VERSION = 'V3'
            self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION,
                                               f'{self.REC_CHAR_TYPE}_rec')
        if not os.path.exists(self.REC_MODEL_PATH):
            self.MODEL_VERSION = 'V3'
            self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION,
                                               f'{self.REC_CHAR_TYPE}_rec_fast')

        # 语言组映射到共享模型
        if self.REC_CHAR_TYPE in self.LATIN_LANG:
            self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION, f'latin_rec_fast')
        elif self.REC_CHAR_TYPE in self.ARABIC_LANG:
            self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION, f'arabic_rec_fast')
        elif self.REC_CHAR_TYPE in self.CYRILLIC_LANG:
            self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION, f'cyrillic_rec_fast')
        elif self.REC_CHAR_TYPE in self.DEVANAGARI_LANG:
            self.REC_MODEL_PATH = os.path.join(self.MODEL_BASE, self.MODEL_VERSION, f'devanagari_rec_fast')
