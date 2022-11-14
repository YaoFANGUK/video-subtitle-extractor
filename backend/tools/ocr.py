from tools.infer import utility
from tools.infer.predict_system import TextSystem
import config
import importlib


# 加载文本检测+识别模型
class OcrRecogniser:
    def __init__(self):
        # 获取参数对象
        importlib.reload(config)
        self.args = utility.parse_args()
        self.recogniser = self.init_model()

    @staticmethod
    def y_round(y):
        y_min = y + 10 - y % 10
        y_max = y - y % 10
        if abs(y - y_min) < abs(y - y_max):
            return y_min
        else:
            return y_max

    def predict(self, id, image, sub_area):
        detection_box, recognise_result = self.recogniser(image)
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
                if len(lines) < 1:
                    lines.append(self.y_round(i[2]))
                else:
                    if self.y_round(i[2]) not in lines \
                            and self.y_round(i[2]) + 10 not in lines \
                            and self.y_round(i[2]) - 10 not in lines:
                        lines.append(self.y_round(i[2]))
            lines = sorted(lines)

            for i in coordinate_list:
                for j in lines:
                    if abs(j - self.y_round(i[2])) <= 10:
                        i[2] = j

            to_rank_res = list(zip(coordinate_list, recognise_result))
            ranked_res = []
            for line in lines:
                tmp_list = []
                for i in to_rank_res:
                    if i[0][2] == line:
                        tmp_list.append(i)
                # 先根据纵坐标排序
                for k in range(1, len(tmp_list)):
                    for j in range(0, len(tmp_list) - k):
                        if tmp_list[j][0][2] > tmp_list[j + 1][0][2]:
                            print(tmp_list[j][0][2])
                            tmp_list[j], tmp_list[j + 1] = tmp_list[j + 1], tmp_list[j]
                # 再根据横坐标排列
                for l in range(1, len(tmp_list)):
                    for j in range(0, len(tmp_list) - l):
                        if tmp_list[j][0][0] > tmp_list[j + 1][0][0]:
                            tmp_list[j], tmp_list[j + 1] = tmp_list[j + 1], tmp_list[j]
                for m in tmp_list:
                    ranked_res.append(m)
            dt_box = []
            for i in [j[0] for j in ranked_res]:
                dt_box.append([(i[0], i[2]), (i[1], i[2]), (i[1], i[3]), (i[0], i[3])])
            res = [i[1] for i in ranked_res]
            return dt_box, res
        else:
            return detection_box, recognise_result

    def init_model(self):
        self.args.use_gpu = config.USE_GPU
        # 设置文本检测模型路径
        self.args.det_model_dir = config.DET_MODEL_PATH
        # 设置文本识别模型路径
        self.args.rec_model_dir = config.REC_MODEL_PATH
        self.args.rec_char_dict_path = config.DICT_PATH
        self.args.rec_image_shape = config.REC_IMAGE_SHAPE
        # 设置识别文本的类型
        self.args.rec_char_type = config.REC_CHAR_TYPE
        return TextSystem(self.args)

    def get_coordinates(self, dt_box):
        return get_coordinates(dt_box)


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
