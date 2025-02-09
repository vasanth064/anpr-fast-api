import os
import cv2
import asyncio
import re
import json
import shutil
import time
import openai

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Tuple, List, Set

# Set your OpenAI API key (or use an environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the results folder as static so images can be served at /results/*
app.mount("/results", StaticFiles(directory="results"), name="results")

# Regex for Indian number plates (e.g., KA01AB1234)
INDIAN_PLATE_PATTERN = re.compile(
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', re.IGNORECASE)

# Valid Indian state codes (first two letters)
VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HR", "HP", "JK",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN",
    "TS", "TR", "UK", "UP", "WB"
}


def clear_results_folder(folder: str):
    """
    Clear all files and subdirectories in the provided folder.
    If the folder does not exist, create it.
    """
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder)
        print(f"Created results folder: {folder}")


async def process_video(video_file: str, frame_interval: int) -> Tuple[Set[str], Set[str], List[str], List[dict]]:
    """
    Process the video using fastanpr. For every nth frame:
      - Normalize and filter plate texts (using regex and state codes).
      - If at least one NEW valid plate is detected, annotate and save the frame.
      - For each such frame, measure processing latency and record per-plate metrics.

    Returns:
      valid_set: Set of unique valid plate texts.
      invalid_set: Set of plate texts not matching our criteria.
      saved_images: List of filenames (relative to /results) for annotated frames.
      performance_metrics: List of dictionaries (one per saved frame) containing performance data.
    """
    if not os.path.exists(video_file):
        print(f"Video file '{video_file}' not found!")
        return None, None, None, None

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error opening video file '{video_file}'!")
        return None, None, None, None

    from fastanpr import FastANPR
    fast_anpr = FastANPR()

    frame_count = 0
    valid_set = set()
    invalid_set = set()
    unique_valid_plates = set()  # To track already saved valid plates
    saved_images = []          # List of saved annotated image filenames
    performance_metrics = []   # List of dicts with per-frame metrics

    results_folder = "results"
    clear_results_folder(results_folder)
    print(f"Cleared results folder: {results_folder}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            start_time = time.time()
            # Convert frame from BGR to RGB for fastanpr
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = await fast_anpr.run([rgb_frame])
            plates = results[0]
            new_valid_plates_in_frame = []
            plates_metrics = []  # To record per-plate performance metrics

            for plate in plates:
                if plate.rec_text is None:
                    continue
                plate_text = plate.rec_text.strip().upper()
                if INDIAN_PLATE_PATTERN.match(plate_text) and plate_text[:2] in VALID_STATE_CODES:
                    valid_set.add(plate_text)
                    if plate_text not in unique_valid_plates:
                        new_valid_plates_in_frame.append(plate)
                        unique_valid_plates.add(plate_text)
                        # Record detection and recognition confidences if available
                        plates_metrics.append({
                            "plate_text": plate_text,
                            "detection_confidence": plate.det_conf if hasattr(plate, "det_conf") else None,
                            "recognition_confidence": plate.rec_conf if hasattr(plate, "rec_conf") else None
                        })
                else:
                    invalid_set.add(plate_text)

            end_time = time.time()
            latency = end_time - start_time

            if new_valid_plates_in_frame:
                # Annotate frame for each new valid plate
                for plate in new_valid_plates_in_frame:
                    bbox = plate.det_box
                    if bbox and len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, plate.rec_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                filename = f"frame_{frame_count}.jpg"
                frame_path = os.path.join(results_folder, filename)
                cv2.imwrite(frame_path, frame)
                saved_images.append(filename)
                # Record performance metrics for this frame
                performance_metrics.append({
                    "frame": frame_count,
                    "latency": latency,
                    "plates": plates_metrics
                })
                print(
                    f"Saved annotated frame {frame_count} as {filename} with latency {latency:.3f}s")

        frame_count += 1

    cap.release()
    return valid_set, invalid_set, saved_images, performance_metrics


async def get_image_metadata(image_path: str) -> dict:
    """
    Simulate sending the image to an OpenAI LLM to extract metadata.
    This function is optional—if any error occurs, it returns an empty dict.
    """
    prompt = (
        f"Analyze the image located at '{image_path}'. "
        "Pretend you are a vehicle inspector. Provide a JSON response with the following keys: "
        "car_color, car_type (e.g., sedan, hatchback, SUV, etc.), number_plate_type (e.g., own vehicle, taxi), "
        "and additional_details. Only return valid JSON."
    )

    def call_openai():
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Hypothetical model with vision capability.
            messages=[
                {"role": "system",
                    "content": "You are a vehicle inspector who analyzes car images."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    try:
        content = await asyncio.to_thread(call_openai)
        metadata = json.loads(content)
        return metadata
    except Exception as e:
        print(f"Metadata extraction failed for {image_path}: {e}")
        return {}


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=JSONResponse)
async def upload_video(file: UploadFile = File(...), frame_interval: int = Form(30)):
    """
    Accepts a video upload, processes it, and then—for each annotated frame—
    attempts to extract optional metadata via OpenAI and collects performance metrics.
    Returns a JSON object with valid/invalid plates, annotated image URLs, metadata (if any),
    and performance metrics.
    """
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, file.filename)

    # Save the uploaded video file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    valid_set, invalid_set, saved_images, perf_metrics = await process_video(file_path, frame_interval)

    # Remove the uploaded video after processing
    os.remove(file_path)

    metadata_dict = {}
    for filename in saved_images:
        local_image_path = os.path.join("results", filename)
        metadata = await get_image_metadata(local_image_path)
        if metadata:
            metadata_dict[f"/results/{filename}"] = metadata

    # Create a performance metrics dictionary mapping image URL to its metrics.
    metrics_dict = {}
    for i, filename in enumerate(saved_images):
        if i < len(perf_metrics):
            metrics_dict[f"/results/{filename}"] = perf_metrics[i]

    results_json = {
        "valid": sorted(list(valid_set)) if valid_set else [],
        "invalid": sorted(list(invalid_set)) if invalid_set else [],
        "images": [f"/results/{img}" for img in saved_images] if saved_images else [],
        "metadata": metadata_dict,  # may be empty if metadata extraction failed
        "metrics": metrics_dict    # performance metrics per annotated frame
    }
    return results_json

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
