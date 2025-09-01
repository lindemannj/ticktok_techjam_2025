import cv2
import numpy as np
import pytesseract

from . import config
from typing import List, Dict, Tuple


def _pixelate_roi_on_image(img: np.ndarray, bbox: Tuple[int, int, int, int], pixel_size: int = 12):
    """
    Pixelates a specified Region of Interest (ROI) on an image.

    This function takes an image (a NumPy array), a bounding box (bbox)
    defining the ROI, and a pixelation size. It resizes the ROI to a small
    version and then scales it back up using `INTER_NEAREST` interpolation,
    which creates the pixelated effect. The pixelated area is then
    overlaid back onto the original image.
    """
    x, y, w, h = bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return
    small_w = max(1, (x1 - x0) // pixel_size)
    small_h = max(1, (y1 - y0) // pixel_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    img[y0:y1, x0:x1] = pixelated


def detect_faces(image: np.ndarray, scaleFactor: float = 1.1, minNeighbors: int = 5) -> List[Dict]:
    """
    Detects faces in an image using a pre-trained Haarcascade classifier.

    It converts the input image to grayscale and uses the `detectMultiScale`
    method from OpenCV to find potential face regions. The function returns a
    list of dictionaries, where each dictionary contains the bounding box (`bbox`)
    for a detected face.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(24, 24)
    )
    return [{"bbox": (int(x), int(y), int(w), int(h))} for (x, y, w, h) in faces]


def detect_text_boxes(image: np.ndarray, min_conf: int = config.MIN_TEXT_CONF, lang: str = config.TESSERACT_LANG,
                      psm: int = config.TESSERACT_PSM) -> List[Dict]:
    """
    Detects text boxes in an image using Tesseract OCR.

    The function converts the image to grayscale and then uses Tesseract's
    `image_to_data` method to find and analyze text. It filters the results
    based on a minimum confidence score (`min_conf`) and returns a list of
    dictionaries. Each dictionary includes the detected text, its confidence,
    and its bounding box.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    config = f"--oem 3 --psm {psm}"
    try:
        data = pytesseract.image_to_data(gray, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    except Exception as e:
        raise RuntimeError(f"Tesseract OCR failed: {e}")

    results = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf >= min_conf:
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            results.append({"text": txt, "conf": conf, "bbox": (x, y, w, h)})
    return results


def process_image_bytes_and_pixelate(image_bytes: bytes,
                                     min_text_conf: int = config.MIN_TEXT_CONF,
                                     text_pixel_size: int = config.TEXT_PIXEL_SIZE,
                                     face_pixel_size: int = config.FACE_PIXEL_SIZE,
                                     tesseract_lang: str = config.TESSERACT_LANG,
                                     tesseract_psm: int = config.TESSERACT_PSM) -> Tuple[bytes, Dict]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    """
    Main function to process an image from binary data.

    This function is the core of the service. It takes raw image bytes,
    decodes them into an image array, then calls `detect_faces` and
    `detect_text_boxes` to find areas that need to be obscured. It then
    pixelates these regions and encodes the final image back into JPEG bytes.
    The function returns the processed image bytes and a metadata dictionary
    containing the detected bounding boxes.
    """
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes")

    face_boxes = detect_faces(img)
    text_boxes = detect_text_boxes(img, min_conf=min_text_conf, lang=tesseract_lang, psm=tesseract_psm)

    for fb in face_boxes:
        _pixelate_roi_on_image(img, fb["bbox"], pixel_size=face_pixel_size)

    for tb in text_boxes:
        _pixelate_roi_on_image(img, tb["bbox"], pixel_size=text_pixel_size)

    is_success, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not is_success:
        raise RuntimeError("Failed to encode processed image to JPEG")

    jpeg_bytes = buffer.tobytes()
    metadata = {"text_boxes": text_boxes, "face_boxes": face_boxes}
    return jpeg_bytes, metadata