# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import dataclasses
from typing import List, Optional

LU, LD, RD, RU = 0, 1, 2, 3
X, Y = 0, 1


@dataclasses.dataclass
class BoundingBoxInfo:
    text: str
    coords: List[float]
    #coords: List[List[float]]
    ocr_score: float = 0.
    shape_type: Optional[str] = None
    raw_coords: Optional[List[float]] = None
    char_mid_coords: Optional[List[float]] = None
    char_coords: Optional[List[List[float]]] = None
    labels: Optional[List[str]] = None
    pred_labels: Optional[List[str]] = None
    index: Optional[int] = None

    def __post_init__(self):
        assert len(self.coords) == 4
        coords = self.coords
        center_x = sum([c[0] for c in coords]) / len(coords)
        center_y = sum([c[1] for c in coords]) / len(coords)
        left_coords = sorted([c for c in coords if c[0] <= center_x], key=lambda x: x[1])
        right_coords = sorted([c for c in coords if c[0] > center_x], key=lambda x: -x[1])
        coords = left_coords + right_coords

        self.coords = coords
        self.center_x = center_x
        self.center_y = center_y
        if self.char_mid_coords and self.char_coords is None:
            self.char_coords = BoundingBoxInfo.calc_char_coords(self.coords, self.char_mid_coords)

    def calc_y_bias(self, slope):
        self.y_bias = self.center_y - self.center_x * slope

    def calc_x_bias(self, slope):
        self.x_bias = self.coords[LU][1] * slope + self.coords[LU][0]

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

    @classmethod
    def from_str(cls, line, index=None):
        cols = line.strip().split("\t")
        assert len(cols) >= 9

        coords = [float(v) for v in cols[:8]]
        coords = [coords[i:i + 2] for i in range(0, len(coords), 2)]

        ocr_score = float(cols[8])
        text = cols[9] if len(cols) > 9 else ""

        if len(cols) > 10:
            assert len(cols) >= 10 + len(text)
            char_mid_coords = [float(v) for v in cols[10:10 + len(text)]]
        else:
            char_mid_coords = None

        return cls(
            text=text,
            coords=coords,
            ocr_score=ocr_score,
            char_mid_coords=char_mid_coords,
            index=index
        )

    @classmethod
    def from_xfund(cls, row, index=None):
        labels = [w["label"] for w in row["words"]] if "words" in row and "label" in row["words"][0] else None
        pred_labels = [w["pred_label"] for w in row["words"]] if "words" in row and "pred_label" in row["words"][
            0] else None
        char_coords = [w["box"] for w in row["words"]] if "words" in row and "box" in row["words"][0] else None
        return cls(
            text=row["text"],
            coords=row["box_four"],
            labels=labels,
            pred_labels=pred_labels,
            char_coords=char_coords,
            index=index
        )

    @classmethod
    def from_labelme(cls, shape, index=None):
        shape_type = shape["shape_type"]
        raw_coords = None
        assert shape_type in {"polygon", "rectangle"}, f"unkonwn shape type {shape_type}"
        if shape_type == "polygon":
            coords = shape["points"]
            if len(coords) != 4:
                raw_coords = coords
                assert len(coords) > 1
                min_x = min([v[0] for v in coords])
                min_y = min([v[1] for v in coords])
                max_x = max([v[0] for v in coords])
                max_y = max([v[1] for v in coords])
                coords = [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, max_y]]
        elif shape_type == "rectangle":
            (x0, y0), (x1, y1) = shape["points"]
            raw_coords = shape["points"]
            coords = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]

        return cls(
            text=shape["label"],
            coords=coords,
            raw_coords=raw_coords,
            shape_type=shape_type,
            index=index
        )

@dataclasses.dataclass
class PadBoundingBoxInfo:
    text: str

class PicInfo:
    def __init__(self, pic_id, size, bounding_boxes, pic_types=None):
        if size is None:
            w, h = 0, 0
            for bbox in bounding_boxes:
                for coord in bbox.coords:
                    w = max(coord[0], w)
                    h = max(coord[1], h)
            size = [w, h]
        self.pic_id = pic_id
        self.size = size
        self.pic_types = pic_types
        self.bounding_boxes = bounding_boxes

        bounding_boxes = sorted(bounding_boxes, key=lambda b:b.center_y)
        slope = PicInfo.calc_slope(bounding_boxes)
        for b in bounding_boxes:
            b.calc_y_bias(slope)
        bounding_boxes = sorted(bounding_boxes, key=lambda b:b.y_bias)
        row_bounding_boxes = PicInfo.greedy_sort_rows(bounding_boxes)

        least_square_slope = PicInfo.calc_least_square_slope(row_bounding_boxes)
        text_slope = slope if abs(slope) > abs(least_square_slope) else least_square_slope
        for b in bounding_boxes:
            b.calc_y_bias(text_slope)
        bounding_boxes = sorted(bounding_boxes, key=lambda b:b.y_bias)
        row_bounding_boxes = PicInfo.greedy_sort_rows(bounding_boxes)
        self._rows = row_bounding_boxes
        for b in bounding_boxes:
            b.calc_x_bias(text_slope)

    @property
    def rows(self):
        return self._rows

    @property
    def sorted_bounding_boxes(self):
        return [bbox for row in self.rows for bbox in row if bbox.text]

    @staticmethod
    def greedy_sort_rows(bounding_boxes, y_bias_thres=12,
                         row_center_ration_diff_thres=0.7,
                         cross_x_rate_thres=0.7):
        pt = re.compile(r"\w")
        rows = list()
        row = list()
        row_total_height, row_len, row_mean_height = 0, 0, 0
        row_total_y_bias, row_mean_y_bias = 0, -y_bias_thres
        for bbox in bounding_boxes:
            if row_mean_height > 0:
                cond1 = abs(bbox.y_bias - row_mean_y_bias) / row_mean_height <= row_center_ration_diff_thres
            else:
                cond1 = abs(bbox.y_bias - row_mean_y_bias) < y_bias_thres
            max_cross_x_rate = max([PicInfo.calc_cross_rate(bbox.coords[LU][X], bbox.coords[RU][X],  a.coords[LU][X],
                                                            a.coords[RU][X]) for a in row]) if row else -10
            cond2 = max_cross_x_rate <= cross_x_rate_thres

            if cond1 and cond2:
                row.append(bbox)
                row_total_y_bias += bbox.y_bias
                row_mean_y_bias = row_total_y_bias / len(row)
                if pt.search(bbox.text):
                    row_total_height += abs(bbox.coords[LU][Y] - bbox.coords[LD][Y])
                    row_len += 1
                    row_mean_height = row_total_height / row_len
            else:
                if row:
                    rows.append(sorted(row, key=lambda x:x.center_x))
                row = [bbox]
                row_total_y_bias = row_mean_y_bias = bbox.y_bias
                row_total_height, row_len, row_mean_height = 0., 0 , 0.
                if pt.search(bbox.text):
                    row_total_height += abs(bbox.coords[LU][Y] - bbox.coords[LD][Y])
                    row_len += 1
                    row_mean_height = row_total_height / row_len
        rows.append(sorted(row, key=lambda x:x.center_x))

        l1 = len("".join([b.text for b in bounding_boxes]))
        l2 = len("".join([b.text for line in rows for b in line]))
        if l1 != l2:
            pass
        assert len("".join([b.text for b in bounding_boxes])) == len("".join([b.text for line in rows for b in line]))

        return rows

    @staticmethod
    def greedy_sort_columns(bounding_boxes, x_bias_thres=30.):
        columns = list()
        last_x_bias = -x_bias_thres
        column = list()
        for bbox in bounding_boxes:
            if abs(bbox.x_bias - last_x_bias) >= x_bias_thres:
                if column:
                    columns.append(column)
                column = [bbox]
            else:
                column.append(bbox)
            last_x_bias = bbox.x_bias
        return columns

    @staticmethod
    def calc_cross_rate(b1, e1, b2, e2):
        min_len = min((e1 - b1), (e2 - b2))
        cross = min(e1, e2) - max(e2 - b2)
        return 0. if min_len <= 0 else cross/min_len

    @staticmethod
    def calc_slope(bounding_boxes, thres=250):
        box_slopes = list()
        for bbox in bounding_boxes:
            coords = bbox.coords
            x_diff_up = coords[RU][X] - coords[LU][X]
            y_diff_up = coords[RU][Y] - coords[LU][Y]
            x_diff_down = coords[RD][X] - coords[LD][X]
            y_diff_down = coords[RD][Y] - coords[LD][Y]
            if (x_diff_up + x_diff_down) / 2. > thres:
                box_slopes.append((y_diff_up/x_diff_up+y_diff_down/x_diff_down)/2.)
        return sum(box_slopes)/len(box_slopes) if box_slopes else 0.

    @staticmethod
    def calc_least_square_slope(row_bounding_boxes, min_col_thres=4,):
        slopes = list()
        for row in row_bounding_boxes:
            if len(row) < min_col_thres:
                continue
            x = np.asarray([b.center_x for b in row])
            y = np.asarray([b.center_y for b in row])
            m, _ = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
            slopes.append(m)
        return sum(slopes)/len(slopes) if slopes else 0.

    @classmethod
    def from_file(cls, path):
        key = os.path.splitext(os.path.basename(path))[0]
        with open(path) as fi:
            parts = next(fi).split("\t")
            assert len(parts) == 2
            size = [float(v) for v in parts]
            bounding_boxes = [BoundingBoxInfo.from_str(s,i) for i,s in enumerate(fi) if s]
        return cls(
            pic_id=key,
            size=size,
            bounding_boxes=bounding_boxes
        )

    @classmethod
    def from_ocr(cls, ocr, filename):
        lines = ocr.split("\n") if isinstance(ocr, str) else ocr
        first_parts = lines[0].split("\t")
        if len(first_parts) == 2:
            size = [float(v) for v in first_parts]
            it = iter(lines[1:])
        else:
            size = None
            it = iter(lines)
        bounding_boxes = [BoundingBoxInfo.from_str(s, i) for i, s in enumerate(it) if s]

        return cls(pic_id=filename, size=size, bounding_boxes=bounding_boxes)

    @classmethod
    def from_xfund(cls, doc):
        if "img" in doc:
            size = (doc["img"]["width"], doc["img"]["height"])
            if isinstance(doc["document"][0]["box"][0], list):
                max_x = max([max(item["box"][0][0], item["box"][1][0]) for item in doc["document"]])
                max_y = max([max(item["box"][0][1], item["box"][1][1]) for item in doc["document"]])
            else:
                max_x = max([max(item["box"][0], item["box"][2]) for item in doc["document"]])
                max_y = max([max(item["box"][1], item["box"][3]) for item in doc["document"]])
            if max_x > size[0] or max_y > size[1]:
                assert max_x < size[1] and max_y < size[0]
                size = [size[1], size[0]]
        else:
            size = None

        bounding_boxes = [BoundingBoxInfo.from_xfund(item, i) for i, item in enumerate(doc["document"])]
        return cls(pic_id=doc["id"], size=size, bounding_boxes=bounding_boxes)

    @classmethod
    def from_labelme(cls, doc, pic_id=None):
        size = (doc["imageHeight"], doc["imageWidth"])
        bounding_boxes = [BoundingBoxInfo.from_labelme(item, i) for i, item in enumerate(doc["shapes"])]
        return cls(pic_id=pic_id, size=size, bounding_boxes=bounding_boxes)

    @classmethod
    def from_json(cls, doc):
        return cls.from_xfund(doc)

    @staticmethod
    def get_coords(box):
        assert len(box) == 4
        return [[box[0], box[1]], [box[0], box[3]], [box[2], box[3]], [box[2],[box[1]]]]

    def to_line(self):
        rows = self.padded_rows()
        lines = list()
        for row in rows:
            lines.append("".join([v.text for v in row]))
        return lines

    def padded_rows(self):
        pt = re.compile(r"\w")
        lines = list()
        for row in self.rows:
            heights = [abs(b.coords[LU][Y] - b.coords[LD][Y]) for b in row if pt.search(b.text)]
            mean_height = sum(heights)/len(heights) if heights else 0.

            line = list()
            last_x = 0.
            for bbox in row:
                if not bbox.text:
                    continue
                space_count = int((bbox.coords[LD][X] - last_x) / mean_height) if mean_height else 10
                if space_count > 0:
                    line.append(PadBoundingBoxInfo(text=" "*space_count))
                line.append(bbox)
                last_x = bbox.coords[RD][X]
            lines.append(line)
        return lines


