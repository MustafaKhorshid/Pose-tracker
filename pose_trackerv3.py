import os
import sys
import time
import urllib.request

try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"Missing library: {e}")
    print("Install them with:  pip install mediapipe opencv-python numpy")
    print("(Make sure you run this script with the same Python where you installed them.)")
    input("\nPress Enter to close...")
    sys.exit(1)

# Optional: microphone library for snap detection
try:
    import sounddevice as sd
except ImportError:
    sd = None

# ---- Look settings (BGR color) ----
COLOR = (26, 26, 255)      # red
BODY_DOT = 6
BODY_LINE = 4
HAND_DOT = 4
HAND_LINE = 3
VISIBILITY_MIN = 0.5       # hide body points the model isn't sure about

# ---- Snap detection settings ----
SAMPLE_RATE = 44100
BLOCK_SIZE = 512           # ~12 ms per audio block
MIN_LEVEL = 0.08           # minimum sharpness a sound needs to count as a snap
AMBIENT_RATIO = 8.0        # sound must also be this many times louder than background
COOLDOWN_SEC = 1.0         # ignore further snaps for this long after a trigger
REQUIRE_HAND = True        # only accept a snap if a hand was visible recently
HAND_GATE_SEC = 0.7        # how recently a hand must have been seen

# ---- Models (downloaded automatically on first run) ----
MODELS = {
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Body only: shoulders, elbows, wrists, hips, knees, ankles, feet (no face 0-10, no rough hand points 17-22)
BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]
BODY_POINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# 21-point hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
]


class SnapDetector:
    """Listens to the microphone and flags sharp, sudden sounds (finger snaps)."""

    def __init__(self):
        self.pending = False
        self.level = 0.0    
        self.threshold = MIN_LEVEL
        self.ambient = 0.01
        self._last_trigger = 0.0
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        # First difference emphasizes sharp, high-frequency clicks over voices and hum
        sharp = float(np.abs(np.diff(indata[:, 0])).max())
        self.level = sharp
        self.threshold = max(MIN_LEVEL, self.ambient * AMBIENT_RATIO)

        t = time.perf_counter()
        if sharp > self.threshold and (t - self._last_trigger) > COOLDOWN_SEC:
            self._last_trigger = t
            self.pending = True
        else:
            # Only learn the background level from non-snap sound
            self.ambient = 0.98 * self.ambient + 0.02 * sharp

    def start(self):
        if sd is None:
            print("sounddevice is not installed, so snap detection is off.")
            print("Install it with:  pip install sounddevice")
            return False
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=BLOCK_SIZE, callback=self._callback,
            )
            self._stream.start()
            return True
        except Exception as e:
            print(f"Could not open the microphone, so snap detection is off: {e}")
            return False

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
    return {name: os.path.join(MODEL_DIR, name) for name in MODELS}


def make_landmarkers(paths):
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=paths["pose_landmarker_full.task"]),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=paths["hand_landmarker.task"]),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    )
    return pose, hands


def to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_body(img, landmarks):
    h, w = img.shape[:2]
    visible = {i: (landmarks[i].visibility or 0) >= VISIBILITY_MIN for i in BODY_POINTS}
    for a, b in BODY_CONNECTIONS:
        if visible[a] and visible[b]:
            cv2.line(img, to_px(landmarks[a], w, h), to_px(landmarks[b], w, h),
                     COLOR, BODY_LINE, cv2.LINE_AA)
    for i in BODY_POINTS:
        if visible[i]:
            cv2.circle(img, to_px(landmarks[i], w, h), BODY_DOT, COLOR, -1, cv2.LINE_AA)


def draw_hand(img, landmarks):
    h, w = img.shape[:2]
    pts = [to_px(lm, w, h) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], COLOR, HAND_LINE, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, HAND_DOT, COLOR, -1, cv2.LINE_AA)


def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    paths = ensure_models()
    pose, hands = make_landmarkers(paths)

    snap = SnapDetector()
    snap_ok = snap.start()

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"Could not open camera {cam_index}. Try another index: python pose_tracker.py 1")
        snap.stop()
        return

    skeleton_only = False
    show_debug = True
    hand_last_seen = -999.0
    toast_text = ""
    toast_until = 0.0

    start = time.perf_counter()
    last_ts = -1
    prev_time = start
    fps = 0.0

    print("Running. Snap to toggle skeleton mode. 'q' quits, 's' toggles manually, 'd' hides debug.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror like a real mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamps must strictly increase
        ts = int((time.perf_counter() - start) * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        pose_result = pose.detect_for_video(mp_image, ts)
        hand_result = hands.detect_for_video(mp_image, ts)

        t = time.perf_counter()
        if hand_result.hand_landmarks:
            hand_last_seen = t

        # ---- Snap handling ----
        if snap.pending:
            snap.pending = False
            if (not REQUIRE_HAND) or (t - hand_last_seen) < HAND_GATE_SEC:
                skeleton_only = not skeleton_only
                toast_text = "SKELETON MODE" if skeleton_only else "CAMERA MODE"
                toast_until = t + 1.0

        canvas = frame if not skeleton_only else frame * 0  # black background

        for landmarks in pose_result.pose_landmarks:
            draw_body(canvas, landmarks)
        for landmarks in hand_result.hand_landmarks:
            draw_hand(canvas, landmarks)

        # ---- Overlay text ----
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        cv2.putText(canvas, f"{fps:.0f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2, cv2.LINE_AA)

        if show_debug:
            if snap_ok:
                info = f"mic sharpness {snap.level:.2f} / needs {snap.threshold:.2f}"
            else:
                info = "snap detection OFF (press s to toggle manually)"
            cv2.putText(canvas, info, (12, 56), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1, cv2.LINE_AA)

        if now < toast_until:
            cv2.putText(canvas, toast_text, (12, canvas.shape[0] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR, 3, cv2.LINE_AA)

        cv2.imshow("Pose Tracker", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            skeleton_only = not skeleton_only
        if key == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()
    snap.stop()
    pose.close()
    hands.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        input("\nSomething went wrong. Press Enter to close...")
