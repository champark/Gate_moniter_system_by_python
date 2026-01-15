import json
import time
import threading
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import mss
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "captures"
ROI_JSON = BASE_DIR / "rois.json"

# =========================
# Gate names
# =========================
GATE_NAMES = {
    1: "1번 게이트",
    2: "2번 게이트",
    3: "3번 게이트",
    4: "4번 게이트",
}

# =========================
# Beep
# =========================
try:
    import winsound
    def do_beep():
        winsound.Beep(1200, 200)
except Exception:
    def do_beep():
        print("\a", end="")

def safe_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def sanitize_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name.strip().strip(".")

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

def draw_korean_text(bgr_img: np.ndarray, text: str, font_path: str, font_size: int = 24) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size)
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

@dataclass
class RuntimeConfig:
    fps: int = 4
    blur_k: int = 9
    sensitivity_level: int = 6     # 1..10, 10 most sensitive
    consec_frames: int = 2

    use_headlight_filter: bool = True
    ignore_bottom_ratio: float = 0.25
    ignore_top_px: int = 0
    canny1: int = 60
    canny2: int = 180

    save_cooldown_sec: float = 5.0
    extra_shots: int = 2
    extra_shot_interval_sec: float = 1.0

    beep_enabled: bool = True

    font_path: str = r"C:\Windows\Fonts\malgun.ttf"
    font_size: int = 24

class WisenetWatcher(threading.Thread):
    def __init__(self, cfg: RuntimeConfig, cfg_lock: threading.Lock, logq: queue.Queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.cfg_lock = cfg_lock
        self.logq = logq
        self.stop_event = threading.Event()

    def log(self, msg: str):
        try:
            self.logq.put_nowait(msg)
        except Exception:
            pass

    def stop(self):
        self.stop_event.set()

    def compute_params(self):
        with self.cfg_lock:
            lvl = int(self.cfg.sensitivity_level)
        lvl = max(1, min(10, lvl))
        t = (lvl - 1) / 9.0

        motion_ratio = 0.030 + (0.008 - 0.030) * t
        thresh = int(round(35 + (20 - 35) * t))

        luma_jump = 5.0 + (15.0 - 5.0) * t
        edge_ratio_min = 0.004 + (0.001 - 0.004) * t

        return motion_ratio, thresh, luma_jump, edge_ratio_min

    def preprocess_gray(self, bgr: np.ndarray, blur_k: int) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        return gray

    def apply_detection_mask(self, gray: np.ndarray, ignore_top_px: int, ignore_bottom_ratio: float) -> np.ndarray:
        g = gray.copy()
        if ignore_top_px > 0 and g.shape[0] > ignore_top_px:
            g[:ignore_top_px, :] = 0
        if ignore_bottom_ratio > 0:
            h = g.shape[0]
            cut = int(h * (1.0 - ignore_bottom_ratio))
            if 0 < cut < h:
                g[cut:, :] = 0
        return g

    def edge_map(self, gray: np.ndarray, c1: int, c2: int) -> np.ndarray:
        return cv2.Canny(gray, c1, c2)

    def save_image(self, view_idx: int, bgr_img: np.ndarray, ratio: float, event_ts: str, seq: int, font_path: str, font_size: int):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        gate = GATE_NAMES.get(view_idx, f"{view_idx}번 게이트")
        fname = sanitize_filename(f"{event_ts}_{gate}_{seq:02d}.png")
        path = SAVE_DIR / fname

        txt = f"{gate}  ratio={ratio:.4f}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        out = draw_korean_text(bgr_img, txt, font_path=font_path, font_size=font_size)

        ok, buf = cv2.imencode(".png", out)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        path.write_bytes(buf.tobytes())
        return path

    def run(self):
        self.log(f"BASE_DIR = {BASE_DIR}")
        self.log(f"SAVE_DIR = {SAVE_DIR}")
        self.log(f"ROI_JSON = {ROI_JSON}")

        try:
            rois = load_rois(ROI_JSON)
        except Exception as e:
            self.log(f"[FATAL] ROI load failed: {repr(e)}")
            return

        prev_gray = [None] * 4
        prev_edge = [None] * 4
        streak = [0] * 4
        last_saved_at = [0.0] * 4
        pending = [[] for _ in range(4)]  # (due_time, event_ts, seq)

        with mss.mss() as sct:
            self.log("Watching... (use Stop button)")
            while not self.stop_event.is_set():
                loop_start = time.time()

                with self.cfg_lock:
                    fps = int(self.cfg.fps)
                    blur_k = int(self.cfg.blur_k)
                    consec_frames = int(self.cfg.consec_frames)

                    use_hf = bool(self.cfg.use_headlight_filter)
                    ignore_bottom = float(self.cfg.ignore_bottom_ratio)
                    ignore_top = int(self.cfg.ignore_top_px)
                    c1, c2 = int(self.cfg.canny1), int(self.cfg.canny2)

                    cooldown = float(self.cfg.save_cooldown_sec)
                    extra_shots = int(self.cfg.extra_shots)
                    extra_interval = float(self.cfg.extra_shot_interval_sec)

                    beep_enabled = bool(self.cfg.beep_enabled)
                    font_path = str(self.cfg.font_path)
                    font_size = int(self.cfg.font_size)

                motion_ratio, thresh, luma_jump, edge_ratio_min = self.compute_params()
                interval = 1.0 / max(1, fps)

                for i, roi in enumerate(rois):
                    if self.stop_event.is_set():
                        break

                    view_idx = i + 1
                    now = time.time()

                    # pending extra shots
                    if pending[i]:
                        ready = [x for x in pending[i] if x[0] <= now]
                        pending[i] = [x for x in pending[i] if x[0] > now]
                        for _, event_ts, seq in ready:
                            try:
                                shot2 = np.array(sct.grab(roi))
                                bgr2 = cv2.cvtColor(shot2, cv2.COLOR_BGRA2BGR)
                                saved2 = self.save_image(view_idx, bgr2, 0.0, event_ts, seq, font_path, font_size)
                            except Exception as e:
                                self.log(f"  FAILED extra save: {repr(e)}")

                    shot = np.array(sct.grab(roi))
                    bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                    gray = self.preprocess_gray(bgr, blur_k)

                    if prev_gray[i] is None:
                        prev_gray[i] = gray
                        if use_hf:
                            det0 = self.apply_detection_mask(gray, ignore_top, ignore_bottom)
                            prev_edge[i] = self.edge_map(det0, c1, c2)
                        continue

                    cur_det = self.apply_detection_mask(gray, ignore_top, ignore_bottom)
                    prev_det = self.apply_detection_mask(prev_gray[i], ignore_top, ignore_bottom)

                    diff = cv2.absdiff(prev_det, cur_det)
                    _, th = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
                    ratio = float(np.count_nonzero(th)) / float(th.size)

                    if use_hf:
                        luma_delta = float(cur_det.mean() - prev_det.mean())

                        cur_e = self.edge_map(cur_det, c1, c2)
                        if prev_edge[i] is None:
                            prev_edge[i] = cur_e

                        ediff = cv2.absdiff(prev_edge[i], cur_e)
                        edge_ratio = float(np.count_nonzero(ediff)) / float(ediff.size)

                        if luma_delta >= luma_jump and edge_ratio < edge_ratio_min:
                            streak[i] = 0
                            prev_gray[i] = gray
                            prev_edge[i] = cur_e
                            continue

                        prev_edge[i] = cur_e

                    if ratio >= motion_ratio:
                        streak[i] += 1
                    else:
                        streak[i] = 0

                    if streak[i] == consec_frames:
                        ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.log(
                            f"[{ts_human}] 감지: 입구{view_idx} "
                            f"(ratio={ratio:.4f}, sens={self.cfg.sensitivity_level}/10)"
                        )
                        if beep_enabled:
                            do_beep()

                        if now - last_saved_at[i] >= cooldown:
                            event_ts = safe_timestamp()
                            try:
                                saved = self.save_image(view_idx, bgr, ratio, event_ts, seq=0, font_path=font_path, font_size=font_size)
                                last_saved_at[i] = now
                                self.log(f"  saved -> {saved.name}")

                                extra_shots_eff = max(0, min(5, extra_shots))
                                for k in range(1, extra_shots_eff + 1):
                                    due = now + extra_interval * k
                                    pending[i].append((due, event_ts, k))

                            except Exception as e:
                                self.log(f"  FAILED to save: {repr(e)}")

                    prev_gray[i] = gray

                dt = time.time() - loop_start
                sleep_t = interval - dt
                if sleep_t > 0:
                    time.sleep(sleep_t)

        self.log("Stopped.")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wisenet Watch (ROI)")
        self.geometry("900x600")

        self.cfg = RuntimeConfig()
        self.cfg_lock = threading.Lock()
        self.logq = queue.Queue()
        self.watcher = None

        self._build_ui()
        self.after(100, self._poll_logs)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        self.btn_start = ttk.Button(top, text="감시 시작", command=self.start_watch)
        self.btn_stop = ttk.Button(top, text="감시 중단", command=self.stop_watch, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.beep_var = tk.BooleanVar(value=True)
        self.chk_beep = ttk.Checkbutton(top, text="Beep ON", variable=self.beep_var, command=self.apply_settings)
        self.chk_beep.pack(side=tk.LEFT, padx=15)

        ttk.Label(top, text="Extra shots(0~5):").pack(side=tk.LEFT, padx=(20, 5))
        self.extra_var = tk.IntVar(value=self.cfg.extra_shots)
        self.spin_extra = ttk.Spinbox(top, from_=0, to=5, textvariable=self.extra_var, width=3,
                                      command=self.apply_settings)
        self.spin_extra.pack(side=tk.LEFT)

        ttk.Label(top, text="interval(s):").pack(side=tk.LEFT, padx=(10, 5))
        self.extra_interval_var = tk.DoubleVar(value=self.cfg.extra_shot_interval_sec)
        self.spin_interval = ttk.Spinbox(top, from_=0.1, to=2.0, increment=0.1, textvariable=self.extra_interval_var,
                                         width=5, command=self.apply_settings)
        self.spin_interval.pack(side=tk.LEFT)

        mid = ttk.Frame(self, padding=10)
        mid.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(mid, text="빛 예민강도 (1~10, 10이 가장 민감):").pack(side=tk.LEFT)
        self.sens_var = tk.IntVar(value=self.cfg.sensitivity_level)

        self.scale = ttk.Scale(mid, from_=1, to=10, orient=tk.HORIZONTAL, command=self._on_scale)
        self.scale.set(self.cfg.sensitivity_level)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # ✅ 반드시 self.sens_label로 만들기
        self.sens_label = ttk.Label(mid, text=f"{self.cfg.sensitivity_level}/10")
        self.sens_label.pack(side=tk.LEFT)

        info = ttk.Frame(self, padding=(10, 0, 10, 10))
        info.pack(side=tk.TOP, fill=tk.X)
        self.params_label = ttk.Label(info, text="")
        self.params_label.pack(side=tk.LEFT)

        bottom = ttk.Frame(self, padding=10)
        bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.txt = tk.Text(bottom, wrap=tk.NONE)
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(bottom, orient=tk.VERTICAL, command=self.txt.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt.config(yscrollcommand=scroll_y.set)

        # ✅ 여기서 한 번만 호출 (sens_label 생성 이후)
        self.apply_settings()

    def _on_scale(self, _):
        lvl = int(round(float(self.scale.get())))
        self.sens_var.set(lvl)
        # sens_label이 있는 상태에서만 업데이트
        if hasattr(self, "sens_label"):
            self.sens_label.config(text=f"{lvl}/10")
        self.apply_settings()

    def apply_settings(self):
        with self.cfg_lock:
            self.cfg.beep_enabled = bool(self.beep_var.get())
            self.cfg.extra_shots = int(self.extra_var.get())
            self.cfg.extra_shot_interval_sec = float(self.extra_interval_var.get())

            lvl = int(self.sens_var.get())
            lvl = max(1, min(10, lvl))
            self.cfg.sensitivity_level = lvl

        # UI가 아직 완성 전일 수도 있으니 안전하게
        if hasattr(self, "params_label"):
            motion_ratio, thresh, luma_jump, edge_ratio_min = self._mapped_params_for_ui(lvl)
            self.params_label.config(
                text=f"MR≈{motion_ratio:.3f}, TH={thresh}, LUMA_JUMP≈{luma_jump:.1f}, EDGE_MIN≈{edge_ratio_min:.4f}"
            )

        if hasattr(self, "chk_beep"):
            self.chk_beep.config(text=("Beep ON" if self.beep_var.get() else "Beep OFF"))

    def _mapped_params_for_ui(self, lvl: int):
        lvl = max(1, min(10, int(lvl)))
        t = (lvl - 1) / 9.0
        motion_ratio = 0.030 + (0.008 - 0.030) * t
        thresh = int(round(35 + (20 - 35) * t))
        luma_jump = 5.0 + (15.0 - 5.0) * t
        edge_ratio_min = 0.004 + (0.001 - 0.004) * t
        return motion_ratio, thresh, luma_jump, edge_ratio_min

    def start_watch(self):
        if not ROI_JSON.exists():
            messagebox.showerror("Error", f"rois.json not found:\n{ROI_JSON}")
            return

        self.apply_settings()

        self.watcher = WisenetWatcher(self.cfg, self.cfg_lock, self.logq)
        self.watcher.start()

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._log_ui("=== START ===")

    def stop_watch(self):
        if self.watcher is not None:
            self.watcher.stop()
            self.watcher = None

        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self._log_ui("=== STOP requested ===")

    def _log_ui(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _poll_logs(self):
        while True:
            try:
                msg = self.logq.get_nowait()
            except queue.Empty:
                break
            self._log_ui(msg)
        self.after(100, self._poll_logs)

    def on_close(self):
        try:
            if self.watcher is not None:
                self.watcher.stop()
        except Exception:
            pass
        self.destroy()

if __name__ == "__main__":
    App().mainloop()
