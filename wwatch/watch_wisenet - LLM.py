# watch_wisenet.py
# - 4 ROI motion detection from rois.json
# - headlight-robust filter (global brightness jump + low edge change => ignore)
# - on trigger: save NOW + save 2 more shots at +0.5s, +1.0s (fresh captures)
# - filename: YYYYMMDD_HHMMSS_1번 게이트_00.png (00=first, 01/02=extra)
# - Korean overlay text drawn with Pillow (Malgun Gothic)
# - Unicode-safe saving via cv2.imencode + Path.write_bytes

import json
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import mss
import numpy as np
import cv2

# ===== paths pinned to script folder =====
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "captures"
ROI_JSON = BASE_DIR / "rois.json"

# ===== beep =====
try:
    import winsound
    def beep():
        winsound.Beep(1200, 200)
except Exception:
    def beep():
        print("\a", end="")

# ===== gate names =====
GATE_NAMES = {
    1: "1번 게이트",
    2: "2번 게이트",
    3: "3번 게이트",
    4: "4번 게이트",
}

def safe_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def sanitize_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name.strip().strip(".")

# ===== detection tuning =====
FPS = 4
BLUR_K = 9              # odd
THRESH = 25
MOTION_RATIO = 0.015
CONSEC_FRAMES = 2

# ===== headlight filter =====
USE_HEADLIGHT_FILTER = True
IGNORE_BOTTOM_RATIO = 0.25   # ignore bottom part only for detection
IGNORE_TOP_PX = 0            # ignore top OSD only for detection

LUMA_JUMP = 8.0
EDGE_RATIO_MIN = 0.0020
CANNY1, CANNY2 = 60, 180

# ===== saving tuning =====
SAVE_COOLDOWN_SEC = 5
EXTRA_SHOTS = 2
EXTRA_SHOT_INTERVAL_SEC = 0.5

# overlay settings
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_SIZE = 24

def load_rois(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        rois = json.load(f)
    if not isinstance(rois, list) or len(rois) != 4:
        raise ValueError("rois.json must contain a list of 4 ROI dicts.")
    for r in rois:
        for k in ("left", "top", "width", "height"):
            if k not in r:
                raise ValueError(f"ROI missing key '{k}': {r}")
    return rois

def preprocess_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
    return gray

def apply_detection_mask(gray: np.ndarray) -> np.ndarray:
    g = gray.copy()

    if IGNORE_TOP_PX > 0 and g.shape[0] > IGNORE_TOP_PX:
        g[:IGNORE_TOP_PX, :] = 0

    if IGNORE_BOTTOM_RATIO > 0:
        h = g.shape[0]
        cut = int(h * (1.0 - IGNORE_BOTTOM_RATIO))
        if 0 < cut < h:
            g[cut:, :] = 0

    return g

def edge_map(gray_for_edges: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray_for_edges, CANNY1, CANNY2)

def draw_korean_text(bgr_img: np.ndarray, text: str) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    x, y = 10, 10
    outline = 2
    for dx in range(-outline, outline + 1):
        for dy in range(-outline, outline + 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 0))

    out_rgb = np.array(img)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

def save_image(view_idx: int, bgr_img: np.ndarray, ratio: float, event_ts: str, seq: int) -> Path:
    """
    seq: 0=trigger frame, 1..EXTRA_SHOTS=extra frames
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    gate = GATE_NAMES.get(view_idx, f"{view_idx}번 게이트")

    # 파일명: YYYYMMDD_HHMMSS_1번 게이트_00.png
    fname = sanitize_filename(f"{event_ts}_{gate}_{seq:02d}.png")
    path = SAVE_DIR / fname

    txt = f"{gate}  ratio={ratio:.4f}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    out = draw_korean_text(bgr_img, txt)

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    path.write_bytes(buf.tobytes())

    return path

def main():
    print("BASE_DIR =", BASE_DIR)
    print("SAVE_DIR =", SAVE_DIR)
    print("ROI_JSON =", ROI_JSON)

    rois = load_rois(ROI_JSON)

    prev_gray = [None] * 4
    prev_edge = [None] * 4
    streak = [0] * 4
    last_saved_at = [0.0] * 4

    # 각 뷰별로 "추가샷 due_time" 리스트 (due_time, event_ts, seq)
    pending = [[] for _ in range(4)]

    interval = 1.0 / FPS

    with mss.mss() as sct:
        print("Watching... (Ctrl+C to stop)")
        while True:
            t0 = time.time()

            for i, roi in enumerate(rois):
                view_idx = i + 1
                now = time.time()

                # ---- process pending extra shots FIRST (fresh capture) ----
                if pending[i]:
                    ready = [x for x in pending[i] if x[0] <= now]
                    pending[i] = [x for x in pending[i] if x[0] > now]
                    for _, event_ts, seq in ready:
                        try:
                            shot2 = np.array(sct.grab(roi))
                            bgr2 = cv2.cvtColor(shot2, cv2.COLOR_BGRA2BGR)
                            saved2 = save_image(view_idx, bgr2, 0.0, event_ts, seq)
                            print(f"  extra saved -> {saved2.name}")
                        except Exception as e:
                            print("  FAILED extra save:", repr(e))

                # ---- normal capture ----
                shot = np.array(sct.grab(roi))  # BGRA
                bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                gray = preprocess_gray(bgr)

                if prev_gray[i] is None:
                    prev_gray[i] = gray
                    if USE_HEADLIGHT_FILTER:
                        prev_edge[i] = edge_map(apply_detection_mask(gray))
                    continue

                cur_det = apply_detection_mask(gray)
                prev_det = apply_detection_mask(prev_gray[i])

                diff = cv2.absdiff(prev_det, cur_det)
                _, th = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)
                ratio = np.count_nonzero(th) / th.size

                # ---- headlight filter ----
                if USE_HEADLIGHT_FILTER:
                    luma_delta = float(cur_det.mean() - prev_det.mean())

                    cur_e = edge_map(cur_det)
                    if prev_edge[i] is None:
                        prev_edge[i] = cur_e

                    ediff = cv2.absdiff(prev_edge[i], cur_e)
                    edge_ratio = np.count_nonzero(ediff) / ediff.size

                    if luma_delta >= LUMA_JUMP and edge_ratio < EDGE_RATIO_MIN:
                        streak[i] = 0
                        prev_gray[i] = gray
                        prev_edge[i] = cur_e
                        continue

                    prev_edge[i] = cur_e

                # ---- streak ----
                if ratio >= MOTION_RATIO:
                    streak[i] += 1
                else:
                    streak[i] = 0

                # ---- trigger ----
                if streak[i] == CONSEC_FRAMES:
                    ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{ts_human}] 동작감지 : 주차장 입구{view_idx} (ratio={ratio:.4f})")
                    beep()

                    if now - last_saved_at[i] >= SAVE_COOLDOWN_SEC:
                        event_ts = safe_timestamp()
                        try:
                            saved = save_image(view_idx, bgr, ratio, event_ts, seq=0)
                            last_saved_at[i] = now
                            print(f"  saved -> {saved.name}")

                            # schedule extra shots (fresh captures)
                            for k in range(1, EXTRA_SHOTS + 1):
                                due = now + EXTRA_SHOT_INTERVAL_SEC * k
                                pending[i].append((due, event_ts, k))

                        except Exception as e:
                            print("  FAILED to save:", repr(e))

                prev_gray[i] = gray

            dt = time.time() - t0
            sleep_t = interval - dt
            if sleep_t > 0:
                time.sleep(sleep_t)

if __name__ == "__main__":
    main()
