import os
import time
import uuid
import shutil
from collections import defaultdict
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO

app = FastAPI(
    title="Pipe Defect Inspection Vision API",
    description="Détection temps réel & Tracking ByteTrack sur canalisations (ONNX Runtime)",
    version="1.0.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.onnx")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH, task="detect")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH,
        "classes": model.names
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), conf: float = 0.35):
    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start_time = time.perf_counter()
    results = model.predict(source=temp_path, conf=conf, imgsz=640, verbose=False)
    res = next(iter(results))
    inference_ms = round((time.perf_counter() - start_time) * 1000, 2)

    detections = []
    for box in res.boxes:
        cls_id = int(box.cls)
        detections.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": round(float(box.conf), 3),
            "bbox_xyxy":  [round(float(coord), 1) for coord in box.xyxy.squeeze().tolist()]
        })

    os.remove(temp_path)
    return {
        "filename": file.filename,
        "inference_time_ms": inference_ms,
        "defects_count": len(detections),
        "detections": detections
    }


@app.post("/simulate/video")
async def simulate_video(
    file: UploadFile = File(...),
    conf: float = 0.35,
    min_frames_alert: int = 5
):
    job_id = str(uuid.uuid4())[:8]
    input_path = f"/tmp/{job_id}_{file.filename}"
    output_path = os.path.join(OUTPUT_DIR, f"tracked_{job_id}.mp4")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Impossible de lire cette vidéo.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    track_history = defaultdict(lambda: {"class": "", "frames_seen": 0, "max_conf": 0.0, "first_frame": -1})
    frame_idx = 0
    start_total = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, imgsz=640, verbose=False)
        res = next(iter(results))

        if res.boxes.id is not None:
            for obj_id, cls_id, c in zip(
                res.boxes.id.int().cpu().tolist(),
                res.boxes.cls.int().cpu().tolist(),
                res.boxes.conf.float().cpu().tolist()
            ):
                info = track_history[obj_id]
                info["class"] = model.names[cls_id]
                info["frames_seen"] += 1
                info["max_conf"] = max(info["max_conf"], round(c, 3))
                if info["first_frame"] == -1:
                    info["first_frame"] = frame_idx

        writer.write(res.plot())

    cap.release()
    writer.release()
    os.remove(input_path)

    total_sec = round(time.perf_counter() - start_total, 2)
    confirmed_alerts = {
        f"ID_{tid}": data
        for tid, data in track_history.items()
        if data["frames_seen"] >= min_frames_alert
    }

    return {
        "job_id": job_id,
        "total_frames": frame_idx,
        "processing_time_sec": total_sec,
        "effective_fps": round(frame_idx / total_sec, 1) if total_sec > 0 else 0,
        "confirmed_unique_defects": len(confirmed_alerts),
        "alerts_log": confirmed_alerts,
        "download_url": f"/download/{job_id}"
    }


@app.get("/download/{job_id}")
def download_video(job_id: str):
    video_path = os.path.join(OUTPUT_DIR, f"tracked_{job_id}.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Vidéo introuvable.")
    return FileResponse(video_path, media_type="video/mp4", filename=f"inspection_{job_id}.mp4")
