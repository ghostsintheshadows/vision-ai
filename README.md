# Vision AI

Real-time Color Detection, Face Detection, and OCR Text Detection — powered by OpenCV and Python. Runs entirely on your own machine, using your own webcam. No cloud, no API key.

## Features

- **Color Detection** — live color tracking with bounding boxes, streamed via MJPEG at ~30fps
- **Face Detection** — real-time face detection using a Haar cascade classifier, with adjustable sensitivity
- **OCR Text Detection** — upload an image and extract text with confidence scores via Tesseract

## Requirements

- Python 3.9+
- A webcam (for the color and face detection features)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed separately (for the text detection feature)

## Setup

1. **Clone this repo**
   ```bash
   git clone https://github.com/YOUR-USERNAME/vision-ai.git
   cd vision-ai
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract** (required for OCR; the app still runs without it, but text detection will be disabled)
   - **macOS:** `brew install tesseract`
   - **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
   - **Windows:** [download the installer](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Run the app**
   ```bash
   python app.py
   ```

5. **Open your browser** at [http://localhost:5000](http://localhost:5000)

## Project structure

```
vision-ai/
├── app.py                  # Flask backend — serves pages + MJPEG streams + API
├── requirements.txt
├── README.md
├── .gitignore
└── static/
    ├── index.html           # Landing page
    ├── color.html           # Color detection page
    ├── face.html            # Face detection page
    ├── text.html            # OCR text detection page
    └── images/              # Static images used by the pages
```

## How it works

Flask serves the pages and streams annotated webcam frames via MJPEG. All video capture and image processing (OpenCV, Tesseract) runs on **your local machine** — the app opens your device's own webcam, so it must be cloned and run locally rather than accessed as a hosted website.

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/stream/color` | GET | MJPEG stream — color detection |
| `/stream/face` | GET | MJPEG stream — face detection |
| `/stream/text` | GET | MJPEG stream — OCR result image |
| `/api/color` | POST | Set the target tracking color |
| `/api/upload` | POST | Upload an image for OCR |
| `/api/face_params` | POST | Adjust face detection sensitivity |
| `/api/face_count` | GET | Current detected face count |
| `/api/ocr_result` | GET | Extracted OCR words + confidence |
| `/api/status` | GET | App status (Tesseract availability, etc.) |

## License

Open source — feel free to fork, modify, and use.
