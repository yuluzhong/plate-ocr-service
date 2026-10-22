import string
import unicodedata
import os
from pathlib import Path
import cv2
import numpy as np


PADDLE_REC_MODEL = os.environ.get('PADDLE_REC_MODEL', 'PP-OCRv5_server_rec')
PADDLE_DEVICE = os.environ.get('PADDLE_DEVICE') or None
OCR_DEBUG = os.environ.get('OCR_DEBUG', '0') == '1'
BASE_DIR = Path(__file__).resolve().parent
PADDLE_REC_MODEL_DIR=str(BASE_DIR / "PP-OCRv5_server_rec_infer")

CHINESE_PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
SPECIAL_PLATE_SUFFIXES = '警学港澳使领挂'
PLATE_CHARS = CHINESE_PROVINCES + SPECIAL_PLATE_SUFFIXES + string.ascii_uppercase + string.digits
_reader = None


def get_ocr_reader():
    global _reader
    if _reader is not None:
        return _reader

    # PaddleOCR downloads official models from HuggingFace by default; BOS is more reliable in China.
    os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE', 'BOS')

    try:
        from paddleocr import TextRecognition
    except ImportError as exc:
        raise RuntimeError(
            'PaddleOCR is not installed. Run "pip install -r requirements.txt" first. '
            'For GPU inference, install the matching paddlepaddle-gpu package instead of CPU paddlepaddle.'
        ) from exc

    try:
        kwargs = {'model_name': PADDLE_REC_MODEL,'model_dir': PADDLE_REC_MODEL_DIR,}
        if PADDLE_DEVICE:
            kwargs['device'] = PADDLE_DEVICE
        _reader = TextRecognition(**kwargs)
    except Exception as exc:
        device_hint = (
            f'Current PADDLE_DEVICE={PADDLE_DEVICE!r}. '
            'Unset it for PaddleOCR auto device selection, or set PADDLE_DEVICE=cpu.'
            if PADDLE_DEVICE else
            'PaddleOCR is using its default device selection. Set PADDLE_DEVICE=cpu if GPU init fails.'
        )
        raise RuntimeError(
            f'PaddleOCR failed to initialize custom model: '
            f'model_name={PADDLE_REC_MODEL}, model_dir={PADDLE_REC_MODEL_DIR}. {device_hint}'
        ) from exc

    return _reader


def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    """
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                row = results[frame_nmr][car_id]
                if 'car' not in row:
                    continue

                car_bbox = row['car']['bbox']
                license_plate = row.get('license_plate', {})
                license_plate_bbox = license_plate.get('bbox', [0, 0, 0, 0])

                f.write('{},{},{},{},{},{},{}\n'.format(
                    frame_nmr,
                    car_id,
                    '[{} {} {} {}]'.format(
                        car_bbox[0],
                        car_bbox[1],
                        car_bbox[2],
                        car_bbox[3]),
                    '[{} {} {} {}]'.format(
                        license_plate_bbox[0],
                        license_plate_bbox[1],
                        license_plate_bbox[2],
                        license_plate_bbox[3]),
                    license_plate.get('bbox_score', 0),
                    license_plate.get('text', '0'),
                    license_plate.get('text_score', 0))
                )


def license_complies_format(text):
    """
    Check if the license plate text complies with the Chinese license plate format.
    """
    if len(text) not in [7, 8]:
        return False

    return text[0] in CHINESE_PROVINCES and \
        text[1] in string.ascii_uppercase and \
        all(char in string.ascii_uppercase + string.digits + SPECIAL_PLATE_SUFFIXES for char in text[2:])


def format_license(text):
    return text


def clean_plate_text(text):
    text = unicodedata.normalize('NFKC', str(text))
    return ''.join(char for char in text.upper() if char in PLATE_CHARS)


def find_plate_like_text(text):
    cleaned = clean_plate_text(text)
    candidates = []
    for start, char in enumerate(cleaned):
        if char not in CHINESE_PROVINCES:
            continue

        for length in (8, 7):
            candidate = cleaned[start:start + length]
            if len(candidate) == length:
                candidates.append(candidate)

    if cleaned:
        candidates.append(cleaned)
    return candidates


def order_points_clockwise(points):
    points = np.asarray(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = points[np.argmin(points.sum(axis=1))]
    rect[2] = points[np.argmax(points.sum(axis=1))]
    rect[1] = points[np.argmin(np.diff(points, axis=1))]
    rect[3] = points[np.argmax(np.diff(points, axis=1))]
    return rect


def points_cover_whole_crop(points, w, h):
    points = order_points_clockwise(points)
    box_w = max(points[:, 0]) - min(points[:, 0])
    box_h = max(points[:, 1]) - min(points[:, 1])
    return box_w > w * 0.94 and box_h > h * 0.94


def contour_to_points(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(contour))


def find_plate_points(image):
    h, w = image.shape[:2]
    contours = []

    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, (25, 35, 35), (105, 255, 255))
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 12), max(3, h // 12))),
            iterations=2,
        )
        contours.extend(cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 10), max(3, h // 3)))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours.extend(cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

    best_points = None
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < w * h * 0.05:
            continue
        points = contour_to_points(contour)
        if points_cover_whole_crop(points, w, h):
            continue
        if area > best_area:
            best_area = area
            best_points = points
    return best_points, len(contours), best_area


def warp_plate_perspective(image, points=None):
    if image is None or image.size == 0:
        if OCR_DEBUG:
            print('warp_plate_perspective: empty image, skip', flush=True)
        return image

    h, w = image.shape[:2]
    if points is None:
        points, contour_count, area = find_plate_points(image)
        if points is None:
            if OCR_DEBUG:
                print(
                    f'warp_plate_perspective: failed, image={w}x{h}, '
                    f'contours={contour_count}, area={area:.1f}',
                    flush=True,
                )
            return image

    points = order_points_clockwise(points)
    if points_cover_whole_crop(points, w, h):
        if OCR_DEBUG:
            rounded_points = np.round(points, 1).tolist()
            print(f'warp_plate_perspective: identity, image={w}x{h}, points={rounded_points}', flush=True)
        return image

    dst_w = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    dst_h = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    if dst_w <= 1 or dst_h <= 1:
        if OCR_DEBUG:
            print(f'warp_plate_perspective: invalid target, image={w}x{h}, target={dst_w}x{dst_h}', flush=True)
        return image

    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, dst), (dst_w, dst_h))
    if warped.shape[0] / max(warped.shape[1], 1) >= 1.5:
        warped = np.rot90(warped)
    if OCR_DEBUG:
        rounded_points = np.round(points, 1).tolist()
        print(
            f'warp_plate_perspective: ok, image={w}x{h}, target={dst_w}x{dst_h}, '
            f'warped={warped.shape[1]}x{warped.shape[0]}, points={rounded_points}',
            flush=True,
        )
    return warped




def result_to_dict(result):
    if isinstance(result, dict):
        return result.get('res', result)
    if hasattr(result, 'res'):
        return result.res
    if hasattr(result, 'json'):
        json_result = result.json
        if isinstance(json_result, dict):
            return json_result.get('res', json_result)
    return {}


def paddle_predict(image, batch_size=1):
    reader = get_ocr_reader()
    try:
        return reader.predict(input=image, batch_size=batch_size)
    except TypeError:
        if isinstance(image, list):
            return [reader.predict(img) for img in image]
        return reader.predict(image)


def extract_paddle_texts(result):
    candidates = []
    for item in result:
        data = result_to_dict(item)

        if data.get('rec_text'):
            score = float(data.get('rec_score') or 0)
            candidates.extend((text, score) for text in find_plate_like_text(data.get('rec_text')))

        texts = data.get('rec_texts') or []
        scores = data.get('rec_scores') or []

        for index, text in enumerate(texts):
            score = scores[index] if index < len(scores) else 0
            candidates.extend((plate_text, float(score)) for plate_text in find_plate_like_text(text))

        joined_texts = find_plate_like_text(''.join(texts))
        if joined_texts:
            joined_score = sum(scores) / len(scores) if scores else 0
            candidates.extend((text, float(joined_score)) for text in joined_texts)

    return candidates


def read_license_plate(license_plate_crop, apply_warp=True):
    """
    Read the license plate text from the given cropped image.
    """
    all_candidates = []

    if apply_warp:
        license_plate_crop = warp_plate_perspective(license_plate_crop)
    result = paddle_predict(license_plate_crop)
    all_candidates.extend(extract_paddle_texts(result))

    if OCR_DEBUG and all_candidates:
        print('OCR candidates:', all_candidates)

    valid_candidates = [
        (format_license(text), score)
        for text, score in all_candidates
        if license_complies_format(text)
    ]
    if valid_candidates:
        return max(valid_candidates, key=lambda item: item[1])

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

