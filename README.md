# Pose Tracker

Real-time body and hand tracking from your webcam, drawn as a red skeleton. Snap your fingers to switch between the normal camera and a skeleton-only view.

Written in Python with OpenCV and MediaPipe. It runs fully on your own machine.

## Features

- Body tracking (shoulders, elbows, wrists, hips, knees, ankles, feet) drawn as red joints and bones
- Face points hidden on purpose, so only the body is shown
- Full hand tracking with 21 points per hand, including every finger joint
- **Snap to toggle:** snap your fingers to switch between camera mode and skeleton-only mode
- Mirrored preview, like a real mirror
- Runs locally. Nothing is uploaded

## Requirements

- Python 3.9 to 3.12 (MediaPipe may not support newer versions yet)
- A webcam
- A microphone (only needed for snap detection)

## Setup

```bash
# 1. Install the libraries
pip install mediapipe opencv-python numpy sounddevice

# 2. Run
python pose_trackerv3.py
```

A virtual environment is optional. If you prefer to keep the libraries separate from your system Python, create one first with `python -m venv venv`, activate it (`venv\Scripts\activate` on Windows, `source venv/bin/activate` on Mac or Linux), and then run the same two steps.

On the first run, the script downloads two MediaPipe model files (about 30 MB) into a `models` folder. After that it works offline.

If the wrong camera opens, pass a camera number: `python pose_trackerv3.py 1`

## Controls

| Key | Action |
|-----|--------|
| `s` | Toggle skeleton mode manually |
| `d` | Show / hide debug info |
| `q` | Quit |

Snap your fingers with a hand in view of the camera to toggle skeleton mode.

## Settings

Edit the constants at the top of `pose_trackerv3.py`:

| Setting | What it does |
|---------|--------------|
| `COLOR` | Skeleton color (BGR order) |
| `BODY_DOT`, `BODY_LINE`, `HAND_DOT`, `HAND_LINE` | Joint and bone sizes |
| `MIN_LEVEL`, `AMBIENT_RATIO` | How sharp a sound must be to count as a snap |
| `COOLDOWN_SEC` | Minimum time between two snap triggers |
| `REQUIRE_HAND` | Only accept a snap when a hand is visible |

For a faster, lighter model, change `pose_landmarker_full` to `pose_landmarker_lite` in the `MODELS` dictionary and in `make_landmarkers`.

## How it works

- **Tracking:** [MediaPipe](https://ai.google.dev/edge/mediapipe) Pose Landmarker (body) and Hand Landmarker (hands) find the joints in each video frame. I use the body model for the main skeleton and the hand model for fingers, because the body model alone only gives rough hand points.
- **Drawing:** OpenCV draws the joints and bones on the video. Face points and the rough body-model hand points are filtered out.
- **Snap detection:** the microphone is read in short blocks. A snap is a very sharp, sudden click, so the code measures how quickly the sound changes and compares it with a slowly adapting background level. A snap only counts if a hand was seen by the camera in the last 0.7 seconds, which reduces false triggers from claps and taps.

## Limitations

- Snap detection is sound-based. Claps, desk taps, and loud noises can also trigger it. Noisy rooms and distant microphones make it less reliable.
- Fast hand movement and overlapping fingers can cause flicker in the hand tracking.
- Tracks one person (body) and up to two hands.

## Credits

Tracking models by [Google MediaPipe](https://ai.google.dev/edge/mediapipe). Built with Python, OpenCV, NumPy, and sounddevice. Written with help from Claude.
