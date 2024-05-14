# -*- coding: utf-8 -*-
import re
import json
import numpy as np
import dataclasses
from typing import List, Optional
from paddleocr import PaddleOCR
from paddleocr.tools.infer import predict_system
from mllm.utils.image_utils import load_image

LU, LD, RD, RU = 0, 1, 2, 3
X, Y = 0, 1

@dataclasses.dataclass
class PadBoundingBoxInfo:
    text: str
@dataclasses.dataclass
class PicInfo:
    def __init__(self, file_id, size, bounding_boxes, pic_types=None):
        if size is None:
            w, h = 0, 0
            for bbox in bounding_boxes:
                for coord in bbox.box_coords:
                    w = max(coord[0], w)
                    h = max(coord[1], h)
            size = [w, h]
        self.file_id = file_id
        self.size = size
        self.pic_types = pic_types
        self.bounding_boxes = bounding_boxes

        bounding_boxes = sorted(bounding_boxes, key=lambda b: b.center_y)
        slope = PicInfo.calc_slope(bounding_boxes)
        for b in bounding_boxes:
            b.calc_y_bias(slope)
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b.y_bias)
        row_bounding_boxes = PicInfo.greedy_sort_rows(bounding_boxes)
        least_square_slope = PicInfo.calc_least_square_slope(row_bounding_boxes)
        text_slope = slope if abs(slope) > abs(least_square_slope) else least_square_slope
        for b in bounding_boxes:
            b.calc_y_bias(text_slope)
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b.y_bias)
        row_bounding_boxes = PicInfo.greedy_sort_rows(bounding_boxes)
        self._rows = row_bounding_boxes
        for b in bounding_boxes:
            b.calc_x_bias(text_slope)

    @property
    def rows(self):
        return self._rows

    @property
    def sorted_bounding_boxes(self):
        return [bbox for row in self.rows for bbox in row if bbox.bbox_text]

    @staticmethod
    def greedy_sort_rows(bounding_boxes,
                         y_bias_thres=12,
                         row_center_ration_diff_thres=0.7,
                         cross_x_rate_thres=0.7):
        pt = re.compile(r"\w")
        rows, row = list(), list()
        row_total_height, row_len, row_mean_height = 0, 0, 0
        row_total_y_bias, row_mean_y_bias = 0, -y_bias_thres
        for bbox in bounding_boxes:
            if row_mean_height > 0:
                cond1 = abs(bbox.y_bias - row_mean_y_bias) / row_mean_height <= row_center_ration_diff_thres
            else:
                cond1 = abs(bbox.y_bias - row_mean_y_bias) < y_bias_thres
            max_cross_x_rate = max([PicInfo.calc_cross_rate(bbox.bbox_coords[LU][X], bbox.bbox_coords[RU][X],
                                                            a.bbox_coords[LU][X], a.bbox_coords[RU][X]) for a in row]) if row else -10
            cond2 = max_cross_x_rate <= cross_x_rate_thres
            if cond1 and cond2:
                row.append(bbox)
                row_total_y_bias += bbox.y_bias
                row_mean_y_bias = row_total_y_bias / len(row)
                if pt.search(bbox.bbox_text):
                    row_total_height += abs(bbox.bbox_coords[LU][Y] - bbox.bbox_coords[LD][Y])
                    row_len += 1
                    row_mean_height = row_total_height / row_len
            else:
                if row:
                    rows.append(sorted(row, key=lambda x: x.center_x))
                row = [bbox]
                row_total_y_bias = row_mean_y_bias = bbox.y_bias
                row_total_height, row_len, row_mean_height = 0., 0, 0.
                if pt.search(bbox.bbox_text):
                    row_total_height += abs(bbox.bbox_coords[LU][Y] - bbox.bbox_coords[LD][Y])
                    row_len += 1
                    row_mean_height = row_total_height / row_len
        rows.append(sorted(row, key=lambda x: x.center_x))
        l1 = len("".join([b.bbox_text for b in bounding_boxes]))
        l2 = len("".join([b.bbox_text for line in rows for b in line]))
        if l1 != l2:
            pass
        assert len("".join([b.bbox_text for b in bounding_boxes])) == len("".join([b.bbox_text for line in rows for b in line]))

        return rows


    @staticmethod
    def greedy_sort_columns():
        pass

    @staticmethod
    def calc_cross_rate(b1, e1, b2, e2):
        min_len = min((e1 - b1), (e2 - b2))
        cross = min(e1, e2) - max(b1, b2)
        return 0. if min_len <= 0 else cross/min_len


    @staticmethod
    def calc_slope(bounding_boxes, thres=250):
        box_slopes = list()
        for bbox in bounding_boxes:
            coords = bbox.bbox_coords
            x_diff_up = coords[RU][X] - coords[LU][X]
            y_diff_up = coords[RU][Y] - coords[LU][Y]
            x_diff_down = coords[RD][X] - coords[LD][X]
            y_diff_down = coords[RD][Y] - coords[LD][Y]
            if (x_diff_up + x_diff_down) / 2. > thres:
                box_slopes.append((y_diff_up / x_diff_up + y_diff_down / x_diff_down) / 2.)
        return sum(box_slopes) / len(box_slopes) if box_slopes else 0.

    @staticmethod
    def calc_least_square_slope(row_bounding_boxes, min_col_thres=4, ):
        slopes = list()
        for row in row_bounding_boxes:
            if len(row) < min_col_thres:
                continue
            x = np.asarray([b.center_x for b in row])
            y = np.asarray([b.center_y for b in row])
            m, _ = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
            slopes.append(m)
        return sum(slopes) / len(slopes) if slopes else 0.

    @classmethod
    def from_ocr_res(cls, ocr_data):
        #ocr_data = json.load(ocr_res)
        file_id = ocr_data["file_id"]
        size = ocr_data["size"] if "size" in ocr_data else None
        bboxes = ocr_data["bboxes"]
        bounding_boxes = [BBoxInfo.from_ocr_box(box, i) for i, box in enumerate(bboxes)]
        return cls(
            file_id=file_id,
            size=size,
            bounding_boxes=bounding_boxes
        )

@dataclasses.dataclass
class BBoxInfo:
    bbox_text: str
    bbox_coords: List[float]
    #bbox_coords: List[List[float]]
    bbox_score: float = 0.
    shape_type: Optional[str] = None
    raw_coords: Optional[List[float]] = None
    char_mid_coords: Optional[List[float]] = None
    char_coords: Optional[List[List[float]]] = None
    labels: Optional[List[str]] = None
    pred_labels: Optional[List[str]] = None
    index: Optional[int] = None

    def __post_init__(self):
        assert len(self.bbox_coords) == 4
        bbox_coords = self.bbox_coords
        center_x = sum([c[0] for c in bbox_coords]) / len(bbox_coords)
        center_y = sum([c[1] for c in bbox_coords]) / len(bbox_coords)
        left_coords = sorted([c for c in bbox_coords if c[0] <= center_x], key=lambda x: x[1])
        right_coords = sorted([c for c in bbox_coords if c[0] > center_x], key=lambda x: -x[1])
        box_coords = left_coords + right_coords

        self.box_coords = box_coords
        self.center_x = center_x
        self.center_y = center_y
        if self.char_mid_coords and self.char_coords is None:
            self.char_coords = BBoxInfo.calc_char_coords(self.bbox_coords, self.char_mid_coords)

    @staticmethod
    def calc_char_coords(coords, char_mid_coords):
        bbox_left_x = (coords[LU][X] + coords[LD][X]) / 2
        bbox_right_x = (coords[RU][X] + coords[RD][X]) / 2
        bbox_up_x_diff = coords[RD][X] - coords[LU][X]
        bbox_down_x_diff = coords[RD][X] - coords[LD][X]
        bbox_up_y_diff = coords[RU][Y] - coords[LU][Y]
        bbox_down_y_diff = coords[RD][Y] - coords[LD][Y]

        char_coords = list()
        for i, char_mid_coord in enumerate(char_mid_coords):
            left_half_width = (char_mid_coord - char_mid_coords[i - 1]) / 2 if i > 0 else char_mid_coord - bbox_left_x
            right_half_width = (char_mid_coords[i + 1] - char_mid_coord) / 2 if i < len(
                char_mid_coords) - 1 else bbox_right_x - char_mid_coord
            half_width = min(abs(left_half_width), abs(right_half_width))

            left_x = max(char_mid_coord - half_width, bbox_left_x)
            right_x = min(char_mid_coord + half_width, bbox_right_x)
            if right_x - left_x < 0.1:
                right_x = left_x = 0.1
            left_up_y = bbox_up_y_diff * (left_x - coords[LU][X]) / bbox_up_x_diff + coords[LU][Y]
            right_up_y = bbox_up_y_diff * (right_x - coords[LU][X]) / bbox_up_x_diff + coords[LU][Y]

            left_down_y = bbox_down_y_diff * (left_x - coords[LD][X]) / bbox_down_x_diff + coords[LD][Y]
            right_down_y = bbox_down_y_diff * (right_x - coords[LD][X]) / bbox_down_x_diff + coords[LD][Y]

            char_coords.append([
                [left_x, left_up_y],
                [left_x, left_down_y],
                [right_x, right_down_y],
                [right_x, right_up_y]
            ])
        return char_coords

    def calc_y_bias(self, slope):
        self.y_bias = self.center_y - self.center_x * slope

    def calc_x_bias(self, slope):
        self.x_bias = self.box_coords[LU][1] * slope + self.box_coords[LU][0]

    @classmethod
    def from_ocr_box(cls, bbox, index=None):
        assert len(bbox["points"]) == 4
        bbox_coords = bbox["points"]
        bbox_text = bbox["transcription"]
        box_score = float(bbox["score"]) if "score" in bbox else 0.
        char_mid_coords = bbox["char_mid_coords"] if "char_mid_coords" in bbox else None

        return cls(
            bbox_text=bbox_text,
            bbox_coords=bbox_coords,
            bbox_score=box_score,
            char_mid_coords=char_mid_coords,
            index=index
        )


def ppocr(image_file: str, ocr: predict_system.TextSystem = None):
    image_data, _ = load_image(image_file, return_chw=False)
    if ocr is None:
        ocr = PaddleOCR(
            use_angle_cls=True,
            use_gpu=True,
            ocr_version="PP-OCRv4",
            det_db_box_thresh=0.1,
            det_db_thresh=0.1,
        )

    text_list = ocr.ocr(image_data, cls=True)
    if text_list[0] is None: return []
    ocr_result = []
    for text in text_list[0]:
        ocr_result.append(
            {
                "transcription": text[1][0],
                "points": text[0],
                "score": text[1][1],
                "difficult": "false"
            }
        )
    # for multiprocessing
    return (image_file, ocr_result)