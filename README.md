# Video Upload Streamlit App

A simple Streamlit application for uploading and previewing video files.

## Features

- Upload video files in multiple formats (MP4, AVI, MOV, MKV, WMV, FLV, WebM)
- Preview uploaded videos directly in the browser
- Display video file information (name, size, type)
- Save uploaded videos to disk
- Clean and intuitive user interface

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## How to Use

1. Click on "Browse files" or drag and drop a video file
2. Preview the video in the browser
3. View video information (filename, size, type)
4. Optionally save the video to the `uploads/` directory

## Supported Video Formats

- MP4
- AVI
- MOV
- MKV
- WMV
- FLV
- WebM

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── uploads/           # Directory for saved videos (created automatically)
└── README.md          # This file
```
