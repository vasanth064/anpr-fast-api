# ANPR System Overview

Below is a high-level overview of how our Automated Number Plate Recognition (ANPR) system operates. We use a FastAPI server to process videos, detect and validate license plates, optionally extract metadata from an LLM, and return the final results (annotated images, performance metrics, etc.) back to the client.

---

## 1. System Architecture

![ANPR System Architecture](https://user-images.githubusercontent.com/.../diagram.png 'System Diagram')

1. **Client**

   - Uploads a video (e.g., `video.mp4`) through the frontend (HTML page) or via a direct API call (cURL, Postman, etc.).
   - Receives the final detection results, including:
     - Valid/invalid plates
     - Annotated frames (if any)
     - Optional metadata from the LLM
     - Performance metrics (latency, detection confidence, etc.)

2. **FastAPI Server**

   - Exposes an `/upload` endpoint that accepts the video file (and parameters like `frame_interval`).
   - Coordinates the entire ANPR pipeline:
     1. Reads the video and splits it into frames at the specified interval.
     2. Passes each frame to the ANPR model for plate detection and OCR-based recognition.
     3. Validates recognized plates against regex and Indian state codes.
     4. Annotates frames for new valid plate detections and saves them.
     5. (Optionally) sends detected images to an LLM to obtain metadata.
   - Aggregates and returns JSON output with the final results.

3. **ANPR Model**

   - Receives frames from the server for plate detection (using YOLOv8 or a similar object detection backbone).
   - Uses OCR (via PaddleOCR or equivalent) to recognize the text in the detected bounding box.
   - Returns detection coordinates, detection confidence, recognized text, and recognition confidence.

4. **LLM (Optional)**
   - Can receive annotated images from the server.
   - Attempts to extract additional metadata, such as:
     - Car color
     - Car type (sedan, hatchback, SUV, etc.)
     - Number plate type (private, taxi, commercial)
     - Any other relevant textual description
   - Sends this metadata back to the server, which merges it into the final JSON response.

---

## 2. Video Processing Pipeline

1. **Upload & Extraction**

   - The client uploads a video (MP4 or another supported format).
   - The server stores it temporarily and reads frames at every _nᵗʰ_ interval (`frame_interval`).

2. **Conversion & Detection**

   - Each selected frame is converted from BGR to RGB.
   - Sent to FastANPR for:
     - **Detection:** Locating the plate’s bounding box.
     - **OCR Recognition:** Extracting the plate text.

3. **Validation & Annotation**

   - The recognized plate text is validated with:
     1. A **regex** check (e.g., `^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$`).
     2. A **state code** check for the first two letters (e.g., `KA`, `DL`, etc.).
   - Only newly detected valid plates are annotated on the frame with bounding boxes and text overlays.
   - Annotated frames are saved in the `results` folder.

4. **Metadata Extraction (Optional)**

   - Each annotated frame can be passed to an LLM for additional insights.
   - If it fails or is turned off, the process continues without the metadata.

5. **Response Construction**
   - The server collects:
     - **Valid** plate texts
     - **Invalid** plate texts
     - Annotated image URLs
     - Optional LLM metadata
     - Performance metrics (latency, detection/recognition confidence)
   - Returns this data as JSON for the client to render.

---

## 3. Validation Layers

1. **Regex Validation**

   - Ensures the plate format follows common Indian license plate conventions (e.g., two letters, two digits, one or two letters, four digits).

2. **State Code Validation**
   - Checks if the first two letters belong to a valid Indian state/territory code (e.g., `KA` for Karnataka, `DL` for Delhi).

Plates failing either check are considered **invalid** and stored in a separate array.

---

## 4. Machine Learning Pipeline

1. **Detection**

   - Built on a YOLOv8 backbone through **fastanpr**. Locates the plate region in the image.

2. **Recognition**

   - An OCR module (PaddleOCR or equivalent) extracts the plate text with a confidence score.

3. **(Optional) LLM Metadata**
   - The system can send the cropped/annotated image to an LLM (GPT-like model) to infer additional details about the car.
   - If the LLM call fails, the pipeline continues without metadata.

---

## 5. Performance Metrics

For every processed frame that results in new valid plates:

- **Latency**: Time taken to run the detection and recognition for that frame.
- **Detection Confidence**: Score from the object detection (YOLOv8).
- **Recognition Confidence**: Score from the OCR text recognition.

These metrics are captured per frame (and per plate) and returned as part of the JSON response.

---

## 6. Testing with cURL

Below is a sample cURL command to upload a video to the `/upload` endpoint, specifying a frame interval of 30:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@/path/to/your/video.mp4" \
  -F "frame_interval=30"
```
