from paddleocr import PaddleOCR
from paddleocr.tools.infer import predict_system
from mllm.utils.image_utils import load_image

def ocr_predict(image_file: str, ocr: predict_system.TextSystem = None):
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