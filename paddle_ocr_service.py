import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile
from typing import List
import uvicorn
from ultralytics import YOLO
import torch

# Import perspective warping and PaddleOCR utilities from project's util.py
from util import warp_plate_perspective, paddle_predict, extract_paddle_texts, clean_plate_text, license_complies_format

app = FastAPI(title="Local PaddleOCR and Plate Detection Service")

# Core directories and device initialization
BASE_DIR = Path(__file__).resolve().parent
device = "cuda" if torch.cuda.is_available() else "cpu"
PLATE_MODEL_PATH = str(BASE_DIR / "best1.pt")

print(f"[OCR Service] Loading plate detection model from: {PLATE_MODEL_PATH} on device: {device}")
plate_model = YOLO(PLATE_MODEL_PATH)

@app.post("/cci_ai/service/v1/plate_number")
async def plate_number(request: Request):
    """
    Receives binary image bytes of a vehicle screenshot, runs YOLOv8 plate detection,
    performs perspective warping (deskewing), runs local PaddleOCR text recognition,
    and returns a JSON response mirroring the client's production API format.
    """
    image_bytes = await request.body()
    
    if not image_bytes:
        return {
            "code": 400,
            "message": "Empty body",
            "result": {"item_list": []}
        }

    # Decode image bytes using OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
   #     print(f"[OCR Service DEBUG] Failed to decode image, bytes_len={len(image_bytes)}")
        return {
            "code": 400,
            "message": "Failed to decode image binary",
            "result": {"item_list": []}
        }
    
  #  print(f"[OCR Service DEBUG] Received image: shape={image.shape}, bytes_len={len(image_bytes)}")

    # 1. Run YOLOv8 OBB plate detector on the vehicle front screenshot
    # Lower conf to 0.20 to capture occluded or blurry plates (e.g. blocked by a bicycle/scooter),
    # while relying on PaddleOCR recognition confidence later to filter out grilles and noise.
    result = plate_model.predict(
        source=image,
        imgsz=960,
        conf=0.25,
        iou=0.70,
        device=device,
        verbose=False
    )[0]

    plate_crop = None
    
    # Check if OBB corners are available
    obb = getattr(result, "obb", None)
    if obb is not None and obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > 0:
        # Sort by confidence score descending and pick the best plate
        scores = obb.conf.cpu().tolist()
        points_list = obb.xyxyxyxy.cpu().numpy()
        best_idx = int(np.argmax(scores))
        best_points = points_list[best_idx].astype(np.float32)
    #    print(f"[OCR Service DEBUG] YOLO OBB detected {len(scores)} plates, best conf={scores[best_idx]:.4f}")
        plate_crop = warp_plate_perspective(image, best_points)
        
    # Fallback to horizontal box if OBB is empty but regular boxes exist
    elif result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.data.cpu().tolist()
        # Sort by confidence score descending
        best_box = max(boxes, key=lambda x: x[4])
        x1, y1, x2, y2 = best_box[:4]
        h, w = image.shape[:2]
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
        plate_crop = image[y1:y2, x1:x2]
   #     print(f"[OCR Service DEBUG] YOLO box detected, best conf={best_box[4]:.4f}")
    # else:
    #     print(f"[OCR Service DEBUG] YOLO detected NO plates at all")
        
    # If no plate crop is found by YOLO plate detector, return empty immediately
    if plate_crop is None or plate_crop.size == 0:
   #     print(f"[OCR Service DEBUG] plate_crop is None or empty, returning empty result")
        return {
            "code": 200,
            "message": "success",
            "result": {
                "item_list": [{"number": "", "type": "", "score": 0.0}]
            }
        }
    
 #   print(f"[OCR Service DEBUG] plate_crop shape={plate_crop.shape}")

    # 2. Run PaddleOCR text recognition on the cropped/warped plate region
    ocr_result = paddle_predict(plate_crop)
    candidates = extract_paddle_texts(ocr_result)
    
 #   print(f"[OCR Service DEBUG] PaddleOCR raw candidates: {candidates}")
    
    # 3. Process candidate texts to find the highest scoring plate text.
    # Enforce a minimum PaddleOCR recognition confidence >= 0.70 to filter out 
    # hallucinated texts on grilles and background noise.
    valid_candidates = []
    for text, score in candidates:
        cleaned = clean_plate_text(text)
        if score >= 0.85 and len(cleaned) >= 2:
            valid_candidates.append((cleaned, score))
        # else:
        #     print(f"[OCR Service DEBUG] Filtered out: text='{text}', cleaned='{cleaned}', score={score:.4f} (need score>=0.85 and len>=2)")
            
    best_number = ""
    best_score = 0.0
    if valid_candidates:
        # 优先选择符合车牌格式的候选（省份+字母+5-6个字符=7或8位）
        # 避免 PaddleOCR 拆分成多段时，高分的部分文字（如 'OP068' score=0.99）
        # 击败正确的完整车牌（如 '皖AOP068' joined score=0.97）
        plate_format_candidates = [
            (text, score) for text, score in valid_candidates
            if license_complies_format(text)
        ]
        if plate_format_candidates:
            best_number, best_score = max(plate_format_candidates, key=lambda x: x[1])
        else:
            best_number, best_score = max(valid_candidates, key=lambda x: x[1])
        
 #   print(f"[OCR Service DEBUG] >>> Final result: number='{best_number}', score={best_score:.4f} <<<")
    
    item_list = [{"number": best_number, "type": "", "score": float(best_score)}]
    
    return {
        "code": 200,
        "message": "success",
        "result": {
            "item_list": item_list
        }
    }

@app.post("/cci_ai/service/v1/plate_number_batch")
async def plate_number_batch(files: List[UploadFile] = File(...)):
    """
    接收上传的多个车辆截图文件，批量进行 YOLOv8 车牌定位和纠偏，
    并使用本地 PaddleOCR 进行批量文本识别，返回批处理识别结果。
    """
    images = []
    valid_original_indices = []
    
    # 1. 批量读取并解码上传的图片数据
    for idx, file in enumerate(files):
        content = await file.read()
        if not content:
            continue
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            valid_original_indices.append(idx)
            
    # 初始化最终返回的批量项列表，默认都为空车牌
    results_list = [{"number": "", "type": "", "score": 0.0} for _ in files]
    
    if not images:
        return {
            "code": 200,
            "message": "success",
            "result": {
                "item_list": results_list
            }
        }
        
    # 2. 批量运行 YOLOv8 OBB 车牌检测器
    yolo_results = plate_model.predict(
        source=images,
        imgsz=960,
        conf=0.25,
        iou=0.70,
        device=device,
        verbose=False
    )
    
    plate_crops = []
    crop_to_original_idx = [] # 记录裁剪图对应 original_idx 也就是 files 中的序号
    
    for i, result in enumerate(yolo_results):
        image = images[i]
        orig_file_idx = valid_original_indices[i]
        plate_crop = None
        
        # 尝试使用 OBB 角点
        obb = getattr(result, "obb", None)
        if obb is not None and obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > 0:
            scores = obb.conf.cpu().tolist()
            points_list = obb.xyxyxyxy.cpu().numpy()
            best_idx = int(np.argmax(scores))
            best_points = points_list[best_idx].astype(np.float32)
            plate_crop = warp_plate_perspective(image, best_points)
        # 降级使用普通矩形框
        elif result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.data.cpu().tolist()
            best_box = max(boxes, key=lambda x: x[4])
            x1, y1, x2, y2 = best_box[:4]
            h, w = image.shape[:2]
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
            plate_crop = image[y1:y2, x1:x2]
            
        if plate_crop is not None and plate_crop.size > 0:
            plate_crops.append(plate_crop)
            crop_to_original_idx.append(orig_file_idx)
            
    # 3. 批量运行 PaddleOCR 识别
    if plate_crops:
        ocr_results = paddle_predict(plate_crops, batch_size=len(plate_crops))
        
        for idx, ocr_res in enumerate(ocr_results):
            orig_file_idx = crop_to_original_idx[idx]
            candidates = extract_paddle_texts(ocr_res)
            
            valid_candidates = []
            for text, score in candidates:
                cleaned = clean_plate_text(text)
                if score >= 0.85 and len(cleaned) >= 2:
                    valid_candidates.append((cleaned, score))
                    
            best_number = ""
            best_score = 0.0
            if valid_candidates:
                # 优先选择符合车牌格式的候选
                plate_format_candidates = [
                    (text, score) for text, score in valid_candidates
                    if license_complies_format(text)
                ]
                if plate_format_candidates:
                    best_number, best_score = max(plate_format_candidates, key=lambda x: x[1])
                else:
                    best_number, best_score = max(valid_candidates, key=lambda x: x[1])
                
            results_list[orig_file_idx] = {"number": best_number, "type": "", "score": float(best_score)}
            
    return {
        "code": 200,
        "message": "success",
        "result": {
            "item_list": results_list
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Plate Detector and Local PaddleOCR service")
    parser.add_argument("--port", type=int, default=9080, help="Port to run the service on")
    parser.add_argument("--host", default="0.0.0.0", help="Host binding IP")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
