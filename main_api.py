import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import tempfile
import requests
import shutil

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch

# Core directories
BASE_DIR = Path(__file__).resolve().parent
ZERO_BBOX = [0.0, 0.0, 0.0, 0.0]
VEHICLE_BORDER_THICKNESS = 5
VEHICLE_BORDER_LINE_LENGTH = 70
CHINESE_PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
# Initialize FastAPI App
app = FastAPI(title="Illegal Parking Capture HTTP Service")

# Find SimHei or other Chinese font for drawing 输出视频用
def find_chinese_font():
    candidates = [
        BASE_DIR / 'fonts' / 'NotoSansCJK-Regular.ttc',
        BASE_DIR / 'fonts' / 'simhei.ttf',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return str(candidate)
    return None

FONT_PATH = find_chinese_font()
#输出视频用
def draw_text(frame, text, x, y, font_size=34):
    text = str(text)
    if not text or text == '0' or text.lower() == 'nan':
        return frame

    if FONT_PATH is None:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, font_size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    x = max(0, min(x, frame.shape[1] - text_width - 12))
    y = max(0, min(y, frame.shape[0] - text_height - 12))
    draw.rectangle((x - 6, y - 6, x + text_width + 6, y + text_height + 6), fill=(255, 255, 255))
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

#输出视频用
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

def bbox_to_str(bbox):
    return '[{} {} {} {}]'.format(*[float(value) for value in bbox])

def clamp(value, low, high):
    return max(low, min(high, value))

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

def bbox_intersection_area(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(float(ax1), float(bx1))
    iy1 = max(float(ay1), float(by1))
    ix2 = min(float(ax2), float(bx2))
    iy2 = min(float(ay2), float(by2))
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

def bbox_iou(bbox_a, bbox_b):
    inter = bbox_intersection_area(bbox_a, bbox_b)
    union = bbox_area(bbox_a) + bbox_area(bbox_b) - inter
    if union <= 0:
        return 0.0
    return inter / union   
def max_overlap_with_other_vehicles(car_bbox, car_id, track_ids):
    own_area = max(bbox_area(car_bbox), 1.0)
    max_iou = 0.0
    max_overlap_ratio = 0.0

    for other_det in track_ids:
        other_id = int(other_det[4])
        if other_id == car_id:
            continue

        other_bbox = other_det[:4]
        inter = bbox_intersection_area(car_bbox, other_bbox)
        if inter <= 0:
            continue

        max_iou = max(max_iou, bbox_iou(car_bbox, other_bbox))
        max_overlap_ratio = max(max_overlap_ratio, inter / own_area)

    return float(max_iou), float(max_overlap_ratio)

def sample_quality_key(sample):
    # Cleaner samples win first; area is only a tie-breaker among similarly clean frames.
    return (
        -float(sample.get("max_vehicle_iou") or 0.0),
        -float(sample.get("max_vehicle_overlap_ratio") or 0.0),
        float(sample.get("vehicle_area") or 0.0),
    )

def keep_best_track_sample(samples, sample, max_samples=71):
    if len(samples) < max_samples:
        samples.append(sample)
        return

    worst_index, worst_sample = min(enumerate(samples), key=lambda item: sample_quality_key(item[1]))
    if sample_quality_key(sample) > sample_quality_key(worst_sample):
        samples[worst_index] = sample
    
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def crop_vehicle_context(frame, car_bbox, target_size=(1024, 768)):
    """
    根据车辆检测框裁剪车辆上下文图。
    适当扩大检测框，避免车牌位于车尾/车头边缘时被裁掉。
    """
    cx1, cy1, cx2, cy2 = car_bbox
    h_frame, w_frame = frame.shape[:2]
    
    cw = cx2 - cx1
    ch = cy2 - cy1
    pad_w = cw * 0.02
    pad_h = ch * 0.02
    
    x1 = clamp(cx1 - pad_w, 0, w_frame - 1)
    y1 = clamp(cy1 - pad_h, 0, h_frame - 1)
    x2 = clamp(cx2 + pad_w, x1 + 1, w_frame)
    y2 = clamp(cy2 + pad_h, y1 + 1, h_frame)
    
    crop = frame[int(y1):int(y2), int(x1):int(x2)]
    
    if crop.size > 0:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    return crop

def call_ocr_api(image_bytes, ocr_api_url, app_key, app_secret):
    """
    Sends binary crop image data to the license plate recognition API endpoint.
    """
    headers = {}
    if app_key:
        headers["App_key"] = app_key
    if app_secret:
        headers["App_secret"] = app_secret
    headers["Content-Type"] = "application/octet-stream"

    try:
        response = requests.post(ocr_api_url, headers=headers, data=image_bytes, timeout=15)
        if response.status_code == 200:
            res_data = response.json()
            if res_data.get("code") == 200:
                item_list = res_data.get("result", {}).get("item_list", [])
                if item_list:
                    # Get the top recognized item
                    plate_number = item_list[0].get("number", "")
                    plate_type = item_list[0].get("type", "")
                    return plate_number, plate_type
            return "", ""
        else:
            print(f"[OCR] API response error: code {response.status_code}, response: {response.text}")
            return "", ""
    except Exception as e:
        print(f"[OCR] Connection failed: {e}")
        return "", ""

# Standard edit distance calculation (Levenshtein distance) in pure Python
def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# Compares if two plate numbers are extremely similar (e.g. edit distance <= 1 character)
def are_plates_similar(p1, p2, max_diff=1):
    c1 = p1
    c2 = p2
    if not c1 or not c2:
        return False
    if abs(len(c1) - len(c2)) > max_diff:
        return False
    return edit_distance(c1, c2) <= max_diff

def is_valid_license_plate(plate_number, plate_type):
    if plate_number[0] not in CHINESE_PROVINCES:
        return False
        
    if len(plate_number) < 2 or plate_number[1] < "A" or plate_number[1] > "Z":
        return False
        
    t = str(plate_type).strip()
    if "绿" in t or "新能源" in t:
        return len(plate_number) == 8
    elif "蓝" in t or "黄" in t or "警" in t or "武警" in t or "教练" in t or "学" in t:
        return len(plate_number) == 7
    else:
        return len(plate_number) in [7, 8]


# Core video processing pipeline
def process_video_pipeline(
    video_path: Path,
    output_dir: Path,
    ocr_api_url: str,
    app_key: str,
    app_secret: str,
    vehicle_weights: str,
    dedup_by_text: bool,
    render_video: bool,
    debug_vehicle:bool,
):
    # Detect PyTorch CUDA GPU device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline] Starting process on video: {video_path} using device: {device}")
    
    # Load YOLO model
    vehicle_model = YOLO(vehicle_weights)
    
    # Open capture video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Pipeline] Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_dist = np.hypot(width / 2, height / 2)

    # 跟踪车辆的行驶状态和样本信息
    # track_id -> { "samples": [ { "frame_idx": int, "car_bbox": list, "vehicle_area": float,  "crop": np_array } ] }
    vehicle_states = {}
    
    # 保存每一帧的完整跟踪历史，用于可选的标注视频渲染
    # frame_idx -> list of { "car_id": int, "car_bbox": list }
    all_frames_data = defaultdict(list)

    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Detect vehicles (COCO classes: car=2, bus=5, truck=7)
        vehicle_class_ids = {2, 5, 7}
        
        # Track vehicles
        # Use Ultralytics built-in ByteTrack
        track_ids = []
        results = vehicle_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.30,
            iou=0.70,
            classes=[2, 5, 7],
            device=device,
            verbose=False
        )[0]
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            for i in range(min(len(boxes), len(ids))):
                x1, y1, x2, y2 = boxes[i][:4]
                car_id = int(ids[i])
                track_ids.append([x1, y1, x2, y2, car_id])
        # Process each tracked vehicle
        for t_det in track_ids:
            x1, y1, x2, y2 = t_det[:4]
            car_id = int(t_det[4])
            car_bbox = [x1, y1, x2, y2]
            
            if car_id not in vehicle_states:
                vehicle_states[car_id] = {
                    "samples": [],
                    "seen_count": 0,
                }
            state = vehicle_states[car_id]
            state["seen_count"] += 1
                
            cw = x2 - x1
            ch = y2 - y1
            area = cw * ch

            # 每个 track 第一次出现必采样，之后按该 track 自己的出现次数每 3 次采样一次。
            should_sample = (state["seen_count"] - 1) % 3 == 0

            # 这里只做非常基础的车辆框合法性检查
            if should_sample and cw > 0 and ch > 0:
                frame_area = max(width * height, 1)
                vehicle_area_ratio = area / frame_area

                # 车辆太小就不送 OCR，太小的车牌基本不可识别
                # 这是高召回里的基础过滤，不是车牌可见性判断
                if vehicle_area_ratio >= 0.01:
                    crop = crop_vehicle_context(frame, car_bbox)

                    if crop is not None and crop.size > 0:
                        samples = state["samples"]
                        max_vehicle_iou, max_vehicle_overlap_ratio = max_overlap_with_other_vehicles(
                            car_bbox,
                            car_id,
                            track_ids,
                        )
                        sample = {
                            "frame_idx": frame_idx,
                            "timestamp_sec": frame_idx / fps,
                            "car_bbox": car_bbox,
                            "vehicle_area": float(area),
                            "vehicle_area_ratio": float(vehicle_area_ratio),
                            "max_vehicle_iou": max_vehicle_iou,
                            "max_vehicle_overlap_ratio": max_vehicle_overlap_ratio,
                            "crop": crop
                        }
                        keep_best_track_sample(samples, sample)
            # 保存用于生成带注释视频的数据
            all_frames_data[frame_idx].append({
                "car_id": car_id,
                "car_bbox": car_bbox
            })

        if frame_idx % 100 == 0:
            print(f"[Pipeline] Frame {frame_idx}/{total_frames} processed.")

    cap.release()

    # 对每个车的样本调用OCR API 
    results_list = []
    print(f"[Pipeline] Processing OCR for {len(vehicle_states)} tracked vehicles.")
    
    temp_crop_dir = Path(output_dir/"debug")
    temp_crop_dir.mkdir(parents=True, exist_ok=True)
    for car_id, state in vehicle_states.items():
        samples = state["samples"]
        if not samples:
#            print(f"[Pipeline DEBUG] car_{car_id}: NO samples collected (seen_count={state['seen_count']}), skipping OCR")
            continue
            
        ocr_candidates = []
        print(f"[Pipeline] Calling OCR API for track ID {car_id} ({len(samples)} samples)...")

        for s_idx, sample in enumerate(samples):
            crop_path = temp_crop_dir / f"car_{car_id}_sample_{s_idx}_frame_{sample['frame_idx']}_{sample['timestamp_sec']:.2f}s.jpg"
            # Write to disk with JPEG quality 80 to ensure size < 100KB
            cv2.imwrite(str(crop_path), sample["crop"], [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            with open(crop_path, "rb") as f:
                img_bytes = f.read()
                
            # Send vehicle crop to OCR service
            plate_number, plate_type = call_ocr_api(img_bytes, ocr_api_url, app_key, app_secret)
            
            if plate_number:
                ocr_candidates.append({
                    "plate_number": plate_number,
                    "plate_type": plate_type,
                    "sample_idx": s_idx
                })
                
        # 清洗、投票，并选择最佳车牌候选结果
        best_plate = ""
        best_type = ""
        best_sample = max(samples, key=lambda s: s["vehicle_area"])
        
        if ocr_candidates:
            # 先过滤：只保留通过合法性校验的车牌参与投票
            valid_ocr_candidates = []
            for cand in ocr_candidates:
                num = cand["plate_number"]
                t = cand["plate_type"]
                if is_valid_license_plate(num, t):
                    valid_ocr_candidates.append(cand)
            if valid_ocr_candidates:
                candidate_counts = defaultdict(int)
                candidate_types = {}
                for cand in valid_ocr_candidates:
                    num = cand["plate_number"]
                    t = cand["plate_type"]
                    candidate_counts[num] += 1
                    if t:
                        candidate_types[num] = t
                        
                best_plate = max(candidate_counts, key=candidate_counts.get)
                best_type = candidate_types.get(best_plate, "")            
            # 选择识别出最佳车牌结果的样本
            matching_samples = [samples[cand["sample_idx"]] for cand in ocr_candidates if cand["plate_number"] == best_plate]
            if matching_samples:
                best_sample = max(matching_samples, key=lambda s: s["vehicle_area"])
                       
        results_list.append({
            "car_id": car_id,
            "plate_number": best_plate,
            "plate_type": best_type,
            "best_frame_idx": best_sample["frame_idx"],
            "timestamp_sec": best_sample["timestamp_sec"],
            "car_bbox": best_sample["car_bbox"],
            "crop_img": best_sample["crop"],
            "area":best_sample["vehicle_area"]
        })
    if not debug_vehicle:
        shutil.rmtree(temp_crop_dir, ignore_errors=True)
    # Step : 根据车牌号相似度聚类，实现不同轨迹之间的去重
    final_records = []
    if dedup_by_text:
        # 按车牌相似度分组（字符编辑距离 ≤ 1）
        clusters = [] 
        '''
        存已经识别出有效车牌的车辆记录，并且按相似车牌分组。 TODO:万一真有编辑距离小于等于的车出现在一个视频里面呢？
        clusters = [
            [
                {"car_id": 3, "plate_number": "皖A12345", ...},
                {"car_id": 8, "plate_number": "皖A12345", ...},
                {"car_id": 9, "plate_number": "皖A1234S", ...},
            ],
            [
                {"car_id": 12, "plate_number": "粤B88888", ...},
            ]
        ]
        '''       
        for record in results_list:
            plate = record["plate_number"]
            if not plate or plate == "0" :
                continue            
            added_to_cluster = False
            for cluster in clusters:
                rep_plate = cluster[0]["plate_number"]
                if are_plates_similar(plate, rep_plate, max_diff=0):
                    cluster.append(record)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([record])
        
        for cluster in clusters:
            best_record = max(cluster, key=lambda item:item["area"] )
            final_records.append(best_record)
    else:
        final_records = results_list

    # Step 5: Save outputs (Crops and CSV report)
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = output_dir / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)
    
    csv_report_path = output_dir / "results.csv"
    report_rows = []

    # 将 track_id 映射到最终车牌号，用于标注视频渲染。
    car_id_to_stable_plate = {}

    for record in final_records:
        car_id = record["car_id"]
        plate = record["plate_number"]
        p_type = record["plate_type"]
        crop_img = record["crop_img"]
        timestamp_sec = record["timestamp_sec"]
         
        # 清理文件名，并追加时间戳，避免文件名冲突，同时补充上下文信息
        safe_type = p_type if p_type else "unknown"
        filename = f"{plate}_{safe_type}_{timestamp_sec:.2f}s.jpg"
        car_id_to_stable_plate[car_id] = f"{plate} ({p_type})" if p_type else plate

        # 将时间戳绘制到截图左上角
        time_str = f"Time: {format_time(timestamp_sec)}"
        cv2.putText(crop_img, time_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(crop_img, time_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        label_text = f"Plate: {plate} ({p_type})" if p_type else f"Plate: {plate}"
        crop_img = draw_text(crop_img, label_text, 20, 72, font_size=34)
        save_path = screenshot_dir / filename
        cv2.imwrite(str(save_path), crop_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        report_rows.append({
            "car_id": car_id,
            "plate_number": plate,
            "plate_type": p_type if p_type else "",
            "screenshot_filename": filename,
            "best_frame": record["best_frame_idx"],
            "timestamp_sec": f"{timestamp_sec:.3f}",
            "formatted_timestamp": format_time(timestamp_sec),
            "car_bbox": bbox_to_str(record["car_bbox"]),
        })

    # Write report CSV
    with open(csv_report_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["car_id", "plate_number", "plate_type", "screenshot_filename", "best_frame", "timestamp_sec", "formatted_timestamp","car_bbox"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"[Pipeline] Finished saving. Report: {csv_report_path}. Screenshots: {len(report_rows)} saved.")

    # Step : 输出视频
    if render_video:
        video_out_path = output_dir / "annotated_evidence.mp4"
        cap = cv2.VideoCapture(str(video_path))
        writer = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        f_idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            f_idx += 1
            
            for row in all_frames_data.get(f_idx, []):
                cid = row["car_id"]
                c_bbox = row["car_bbox"]
                
                # Draw vehicle box boundary
                cx1, cy1, cx2, cy2 = [int(round(v)) for v in c_bbox]
                draw_border(
                    frame,
                    (cx1, cy1),
                    (cx2, cy2),
                    (0, 255, 0),
                    VEHICLE_BORDER_THICKNESS,
                    VEHICLE_BORDER_LINE_LENGTH,
                    VEHICLE_BORDER_LINE_LENGTH
                )
                
                # Draw label if stable text is assigned
                stable_label = car_id_to_stable_plate.get(cid, "Processing")
                display_label = f"[{cid}] {stable_label}"
                if stable_label:
                    frame = draw_text(frame, display_label, cx1, cy1 - 44)
                        
            writer.write(frame)
            
        cap.release()
        writer.release()
        print(f"[Pipeline] Annotated video rendered: {video_out_path}")

    print("[Pipeline] Task completed successfully.")


# Background task runner for FastAPI
def background_processing_task(
    temp_file_path: str,
    output_dir_path: str,
    ocr_api_url: str,
    app_key: str,
    app_secret: str,
    vehicle_weights: str,
    dedup_by_text: bool,
    render_video: bool,
    debug_vehicle:bool,
):
    try:
        video_path = Path(temp_file_path)
        output_dir = Path(output_dir_path)
        process_video_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            ocr_api_url=ocr_api_url,
            app_key=app_key,
            app_secret=app_secret,
            vehicle_weights=vehicle_weights,
            dedup_by_text=dedup_by_text,
            render_video=render_video,
            debug_vehicle=debug_vehicle,
        )
    except Exception as e:
        print(f"[Background Task] Failed to run pipeline: {e}")
    finally:
        # Delete the temporary uploaded video file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# FastAPI HTTP Endpoint definitions
@app.post("/cci_ai/service/v1/capture_parking")
async def capture_parking(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ocr_api_url: str = Form("http://127.0.0.1:9080/cci_ai/service/v1/plate_number"),
    app_key: str = Form(""),
    app_secret: str = Form(""),
    dedup: bool = Form(True), #是否去重
    render: bool = Form(True), #是否取证视频
    debug_vehicle:bool = Form(False)  #是否取证 所有 car_id
):
    """
    接收上传的停车视频，保存后立即返回视频收据状态，并在后台线程中安排车辆检测处理流程。
    """
    print(f"[API] Received illegal parking capture request. Video file: {file.filename}")
    
    # 将上传的文件保存到临时位置
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd)
    
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    video_stem = Path(file.filename).stem
    safe_video_name = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in video_stem
    ).strip("._")

    output_dir = BASE_DIR / "output_results" / (safe_video_name or "uploaded_video")

    # Schedule background processing
    background_tasks.add_task(
        background_processing_task,
        temp_file_path=temp_path,
        output_dir_path=str(output_dir),
        ocr_api_url=ocr_api_url,
        app_key=app_key,
        app_secret=app_secret,
        vehicle_weights=str(BASE_DIR / "yolov8n.pt"),
        dedup_by_text=dedup,
        render_video=render,
        debug_vehicle=debug_vehicle,
    )
    
    # Return transmission/upload status immediately to the robot/caller
    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "message": "Video transmitted successfully. Start processing task.",
            "filename": file.filename,
            "status": "processing",
            "output_results_dir": str(output_dir)
        }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI Web Service for Illegal Parking Capture")
    parser.add_argument("--port", type=int, default=8080, help="Port to run FastAPI on")
    parser.add_argument("--host", default="0.0.0.0", help="Host binding IP")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)


