import os
import cv2
import asyncio
import re
import json
import shutil
import time
import openai
import base64
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Tuple, List, Set

# Set your OpenAI API key (or use an environment variable)
openai.api_key = os.getenv(
    "OPENAI_API_KEY",process.env.OPENAI_API_KEY)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the results folder as static so images can be served at /results/*
app.mount("/results", StaticFiles(directory="results"), name="results")

# Regex for Indian number plates (e.g., KA01AB1234)
INDIAN_PLATE_PATTERN = re.compile(
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', re.IGNORECASE)

# Valid Indian state codes (the first two letters)
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
      - Normalize and filter plate texts (regex + state code).
      - If at least one new valid plate is found, annotate & save that frame.
      - Collect performance metrics (latency, detection confidence, etc.).

    Returns:
      valid_set: Set of unique valid plate texts.
      invalid_set: Set of unique invalid plate texts.
      saved_images: List of saved annotated frame filenames.
      performance_metrics: List of dicts with per-frame metrics.
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
    unique_valid_plates = set()  # Track already saved valid plates
    saved_images = []            # List of annotated frames
    performance_metrics = []     # List of per-frame metric dicts

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
            plates_metrics = []  # Performance details per plate

            for plate in plates:
                if plate.rec_text is None:
                    continue
                plate_text = plate.rec_text.strip().upper()
                # Validate plate format + state code
                if INDIAN_PLATE_PATTERN.match(plate_text) and plate_text[:2] in VALID_STATE_CODES:
                    valid_set.add(plate_text)
                    if plate_text not in unique_valid_plates:
                        new_valid_plates_in_frame.append(plate)
                        unique_valid_plates.add(plate_text)
                        plates_metrics.append({
                            "plate_text": plate_text,
                            "detection_confidence": getattr(plate, "det_conf", None),
                            "recognition_confidence": getattr(plate, "rec_conf", None),
                        })
                else:
                    invalid_set.add(plate_text)

            end_time = time.time()
            latency = end_time - start_time

            # If we have newly detected valid plates, annotate & save
            if new_valid_plates_in_frame:
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
    Demonstration of how to pass an image to OpenAI Vision model
    by embedding it in 'image_url' as a data URL.

    We'll ask the model for JSON describing the car.
    The doc example uses 'gpt-4o-mini', but adjust the model
    to one you actually have access to.
    """
    # 1. Read the image as bytes & base64-encode
    try:
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        print(f"Failed to read image: {e}")
        return {}

    # 2. Construct data URL
    data_url = f"data:image/jpeg;base64,{encoded_image}"

    # 3. Build our ChatCompletion message array
    #    "content" is now a list with text + image blocks
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please provide a JSON response with the keys: car_color, car_type, "
                        "number_plate_type, additional_details. Only return valid JSON as "
                        "plain text only. Do not use markdown formatting."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ],
        }
    ]

    def call_openai():
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # e.g., vision-capable model per your docs
                messages=messages,
                max_tokens=300,
            )
            print(f"OpenAI response: {response}")
            return response["choices"][0]["message"]["content"]

        except Exception as exc:
            print(f"Error calling OpenAI: {exc}")
            return None

    # Run the blocking API call in a thread
    content = await asyncio.to_thread(call_openai)
    print(content, "content")
    if not content:
        return {}

    # Attempt to parse the response as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Received invalid JSON from ChatCompletion.")
        return {}


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=JSONResponse)
async def upload_video(file: UploadFile = File(...), frame_interval: int = Form(30)):
    """
    1. Accept a video file + frame_interval.
    2. Process the video:
       - Use FastANPR to detect & annotate plates.
       - Collect performance metrics (latency, detection/recognition conf).
    3. For each annotated frame, optionally call the vision model to get metadata.
    4. Return a JSON object with:
       - valid plates
       - invalid plates
       - list of annotated image URLs
       - vision metadata for each image (if any)
       - performance metrics
    """
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, file.filename)

    # Save the uploaded video
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    valid_set, invalid_set, saved_images, perf_metrics = await process_video(file_path, frame_interval)
    os.remove(file_path)  # Remove the uploaded file after processing

    # For each annotated image, optionally get metadata
    metadata_dict = {}
    for filename in saved_images:
        local_image_path = os.path.join("results", filename)
        metadata = await get_image_metadata(local_image_path)
        if metadata:
            metadata_dict[f"/results/{filename}"] = metadata

    # Build a mapping of each image -> performance metrics
    metrics_dict = {}
    for i, filename in enumerate(saved_images):
        if i < len(perf_metrics):
            metrics_dict[f"/results/{filename}"] = perf_metrics[i]

    results_json = {
        "valid": sorted(list(valid_set)) if valid_set else [],
        "invalid": sorted(list(invalid_set)) if invalid_set else [],
        "images": [f"/results/{img}" for img in saved_images] if saved_images else [],
        "metadata": metadata_dict,  # Vision-based metadata if successful
        "metrics": metrics_dict     # Performance metrics per frame
    }
    return results_json

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
