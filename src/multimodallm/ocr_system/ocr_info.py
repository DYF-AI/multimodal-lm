# -*- coding: utf-8 -*-
import dataclasses
from typing import List, Optional

LU, LD, RD, RU = 0, 1, 2, 3
X, Y = 0, 1


class PicInfo:
    def __init__(self):
        pass

    @property
    def rows(self):
        pass

    @property
    def sorted_bounding_boxes(self):
        pass

    @staticmethod
    def greedy_sort_rows():
        pass

    @staticmethod
    def greedy_sort_columns():
        pass

    @staticmethod
    def calc_cross_rate(b1, e1, b2, e2):
        pass

    @staticmethod
    def calc_slope(bounding_boxes, thres=250):
        pass

    @staticmethod
    def calc_least_square_slope(row_bounding_boxes, min_col_thres=4, ):
        pass

    @classmethod
    def from_ocr(cls, ocr, filename):
        pass

@dataclasses.dataclass
class BoxInfo:
    box_text: str
    # box_coords: List[float]
    box_coords: List[List[float]]
    box_score: float = 0.
    shape_type: Optional[str] = None
    raw_coords: Optional[List[float]] = None
    char_mid_coords: Optional[List[float]] = None
    char_coords: Optional[List[List[float]]] = None
    labels: Optional[List[str]] = None
    pred_labels: Optional[List[str]] = None
    index: Optional[int] = None

    def __post_init__(self):
        assert len(self.box_coords) == 4
        box_coords = self.box_coords
        center_x = sum([c[0] for c in box_coords]) / len(box_coords)
        center_y = sum([c[1] for c in box_coords]) / len(box_coords)
        left_coords = sorted([c for c in box_coords if c[0] <= center_x], key=lambda x: x[1])
        right_coords = sorted([c for c in box_coords if c[0] > center_x], key=lambda x: -x[1])
        box_coords = left_coords + right_coords

        self.box_coords = box_coords
        self.center_x = center_x
        self.center_y = center_y
        if self.char_mid_coords and self.char_coords is None:
            self.char_coords = BoxInfo.calc_char_coords(self.box_coords, self.char_mid_coords)

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
    def from_ocr_box(cls, box, index=None):
        assert len(box["points"]) == 4
        box_coords = box["point"]
        box_text = box["transcription"]
        box_score = float(box["score"])
        char_mid_coords = box["char_mid_coords"] if "char_mid_coords" in box else None

        return cls(
            box_text=box_text,
            box_coords=box_coords,
            box_score=box_score,
            char_mid_coords=char_mid_coords,
            index=index
        )
