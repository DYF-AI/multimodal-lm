
import os
import difflib


def list_file_mapping(root_path, suffixes=None, no_suffixes=None, is_warp=None):
    root_path = root_path.replace("\\", "/")
    file_mapping,deplicated = dict(), set()
    for dirpath, dirs, files in os.walk(root_path):
        for filename in files:
            if no_suffixes is not None:
                if filename.endswith(no_suffixes):
                    continue
            if suffixes is not None:
                if not filename.endswith(suffixes):
                    continue
            filepath = os.path.join(dirpath, filename).replace("\\", "/")
            fileid = os.path.splitext(filepath.replace(root_path, ""))[0]
            if is_warp:
                fileid = os.path.join(os.path.dirname(fileid), os.path.basename(fileid).split("_")[0])
            if fileid in fileid:
                deplicated.add(fileid)
            file_mapping[fileid] = filepath
    return file_mapping, deplicated

def get_ocr_to_image_mapping(image_path, ocr_path, is_warp=False):
    image_mapping, image_deplicated = list_file_mapping(image_path, no_suffixes=[".txt", ".db"], is_warp=is_warp)
    ocr_mapping, ocr_deplicated = list_file_mapping(ocr_path, no_suffixes=[".txt"], is_warp=is_warp)
    ocr_to_image = dict()
    for fileid, ocr_path in ocr_mapping.items():
        if fileid not in image_mapping or fileid in image_deplicated or fileid in ocr_deplicated:
            continue
        ocr_to_image[ocr_path] = image_mapping[fileid]
    return ocr_to_image


def string_similar(s1, s2):
    if not s1:
        s1 = ""
    if not s2:
        s2 = ""
    assert isinstance(s1 ,str)
    assert isinstance(s2, str)
    return difflib.SequenceMatcher(None, str(s1), str(s2)).quick_ratio()

def sort_and_union_values(values_list, by="图片路径"):
    list_of_dict = list()
    for values in values_list:
        if isinstance(values, dict):
            for k, v in values.items():
                v["__id__"] = k
            list_of_dict.append(values)
            continue

        value_dict = dict()
        for value in values:
            if isinstance(by, str):
                key = value[by]
            elif callable(by):
                key = by(value)
            else:
                if "__id__" in value:
                    key = value["__id__"]
                    value_dict[key] = value
                    continue
                raise Exception(f"unsupported {by}")
            if not key:
                continue
            value["__id__"] = key
            value_dict[key] = value
        list_of_dict.append(value_dict)
    common_keys = list(set.intersection(*[set(d.keys()) for d in list_of_dict]))

    return [
        [value_dict[k] for k in common_keys] for value_dict in list_of_dict
    ]

if __name__ == "__main__":
    pass
