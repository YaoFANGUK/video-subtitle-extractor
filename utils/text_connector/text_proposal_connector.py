import numpy as np

from utils.text_connector.other import clip_boxes
from utils.text_connector.text_proposal_graph_builder import TextProposalGraphBuilder


class TextProposalConnector:
    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        # tp=text proposal
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)
        text_lines = np.zeros((len(tp_groups), 5), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]

            x0 = np.min(text_line_boxes[:, 0])
            x1 = np.max(text_line_boxes[:, 2])

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5

            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)
            text_lines[index, 4] = score

        text_lines = clip_boxes(text_lines, im_size)

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
            text_recs[index, 0] = xmin
            text_recs[index, 1] = ymin
            text_recs[index, 2] = xmax
            text_recs[index, 3] = ymin
            text_recs[index, 4] = xmax
            text_recs[index, 5] = ymax
            text_recs[index, 6] = xmin
            text_recs[index, 7] = ymax
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs
