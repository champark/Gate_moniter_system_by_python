import json
import time
import threading
import queue
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageTk

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
    sensitivity_level: int = 6     # 1..10 (10 most sensitive)
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

    # ✅ 미리보기 표시할 사진 번호: 1=기본샷, 2=extra#1, 3=extra#2 ...
    preview_shot_no: int = 2  # 기본 2번째(=extra#1)

    font_path: str = r"C:\Windows\Fonts\malgun.ttf"
    font_size: int = 24


class WisenetWatcher(threading.Thread):
    """
    감시 스레드.
    - logq  : 텍스트 로그(표시용)
    - eventq: UI 업데이트용 이벤트(dict)
    """
    def __init__(self, cfg: RuntimeConfig, cfg_lock: threading.Lock,
                 logq: queue.Queue, eventq: queue.Queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.cfg_lock = cfg_lock
        self.logq = logq
        self.eventq = eventq
        self.stop_event = threading.Event()

    def log(self, msg: str):
        try:
            self.logq.put_nowait(msg)
        except Exception:
            pass

    def emit(self, event: dict):
        try:
            self.eventq.put_nowait(event)
        except Exception:
            pass

    def stop(self):
        self.stop_event.set()

    def compute_params(self):
        """
        sensitivity_level (1..10), 10=most sensitive
        - motion_ratio: 0.030(lvl1) -> 0.008(lvl10)
        - thresh:      35(lvl1) -> 20(lvl10)
        - headlight filter adjustment with sensitivity:
            luma_jump: 5.0(lvl1) -> 15.0(lvl10)
            edge_min:  0.004(lvl1)-> 0.001(lvl10)
        """
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

    def save_image(self, view_idx: int, bgr_img: np.ndarray, ratio: float,
                   event_ts: str, seq: int, font_path: str, font_size: int):
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

        # pending extra shots: (due_time, event_ts, seq)
        pending = [[] for _ in range(4)]

        self.log("Watching... (use Stop button)")
        self.emit({"type": "status", "running": True})

        with mss.mss() as sct:
            while not self.stop_event.is_set():
                loop_start = time.time()

                # cfg snapshot
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

                    # ---- handle pending extra shots ----
                    if pending[i]:
                        ready = [x for x in pending[i] if x[0] <= now]
                        pending[i] = [x for x in pending[i] if x[0] > now]
                        for _, event_ts, seq in ready:
                            try:
                                shot2 = np.array(sct.grab(roi))
                                bgr2 = cv2.cvtColor(shot2, cv2.COLOR_BGRA2BGR)
                                saved2 = self.save_image(view_idx, bgr2, 0.0, event_ts, seq, font_path, font_size)
                                self.log(f"  extra saved -> {saved2.name}")
                                self.emit({
                                    "type": "saved",
                                    "gate": view_idx,
                                    "filename": saved2.name,
                                    "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "seq": seq,
                                })
                            except Exception as e:
                                self.log(f"  FAILED extra save: {repr(e)}")

                    # ---- normal capture ----
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

                    # ---- headlight filter ----
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

                    # ---- streak ----
                    if ratio >= motion_ratio:
                        streak[i] += 1
                    else:
                        streak[i] = 0

                    # ---- trigger ----
                    if streak[i] == consec_frames:
                        ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.log(f"[{ts_human}] 감지: 입구{view_idx} (ratio={ratio:.4f})")

                        self.emit({
                            "type": "detected",
                            "gate": view_idx,
                            "ratio": ratio,
                            "when": ts_human,
                        })

                        if beep_enabled:
                            do_beep()

                        # ---- save (base + extras) ----
                        if now - last_saved_at[i] >= cooldown:
                            event_ts = safe_timestamp()
                            try:
                                saved = self.save_image(view_idx, bgr, ratio, event_ts, seq=0,
                                                        font_path=font_path, font_size=font_size)
                                last_saved_at[i] = now
                                self.log(f"  saved -> {saved.name}")
                                self.emit({
                                    "type": "saved",
                                    "gate": view_idx,
                                    "filename": saved.name,
                                    "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "seq": 0,
                                })

                                extra_eff = max(0, min(5, extra_shots))
                                for k in range(1, extra_eff + 1):
                                    due = now + extra_interval * k
                                    pending[i].append((due, event_ts, k))

                            except Exception as e:
                                self.log(f"  FAILED to save: {repr(e)}")

                    prev_gray[i] = gray

                dt = time.time() - loop_start
                sleep_t = interval - dt
                if sleep_t > 0:
                    time.sleep(sleep_t)

        self.emit({"type": "status", "running": False})
        self.log("Stopped.")


class App(tk.Tk):
    RECENT_WINDOW_SEC = 8.0  # 최근 감지 표시 유지시간(초)

    def __init__(self):
        super().__init__()
        self.title("Wisenet Watch (ROI)")
        self.geometry("1240x760")

        self.cfg = RuntimeConfig()
        self.cfg_lock = threading.Lock()
        self.logq = queue.Queue()
        self.eventq = queue.Queue()
        self.watcher = None

        # Gate UI state
        self.last_ratio = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self.last_detect_time = {1: "-", 2: "-", 3: "-", 4: "-"}
        self.last_saved_file = {1: "-", 2: "-", 3: "-", 4: "-"}
        self.last_detect_epoch = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

        # 오늘 누적 카운터
        self.count_date = datetime.now().strftime("%Y%m%d")
        self.detect_count = {1: 0, 2: 0, 3: 0, 4: 0}

        self.last_global_saved = "-"
        self.last_global_detected = "-"

        # ✅ gate별 seq별 마지막 저장 파일 기록
        self.saved_by_gate_seq = {1: {}, 2: {}, 3: {}, 4: {}}

        # ✅ 현재 미리보기 게이트
        self.preview_gate = 1
        self.latest_saved_ref = None  # (gate:int, seq:int, filename:str)

        self._build_styles()
        self._build_ui()

        self.after(100, self._poll_logs)
        self.after(100, self._poll_events)
        self.after(500, self._update_capture_count_loop)
        self.after(250, self._update_recent_colors_loop)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------
    # Styles
    # -------------------------
    def _build_styles(self):
        style = ttk.Style(self)
        style.configure("Recent.TLabel", foreground="green")
        style.configure("Old.TLabel", foreground="gray")

    # -------------------------
    # safe parsers
    # -------------------------
    def _safe_int(self, v, default: int, min_v=None, max_v=None) -> int:
        try:
            x = int(v)
        except Exception:
            x = default
        if min_v is not None:
            x = max(min_v, x)
        if max_v is not None:
            x = min(max_v, x)
        return x

    def _safe_float(self, v, default: float, min_v=None, max_v=None) -> float:
        try:
            x = float(v)
        except Exception:
            x = default
        if min_v is not None:
            x = max(min_v, x)
        if max_v is not None:
            x = min(max_v, x)
        return x

    # -------------------------
    # UI build
    # -------------------------
    def _build_ui(self):
        # ===== TOP BAR =====
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        self.btn_start = ttk.Button(top, text="감시 시작", command=self.start_watch)
        self.btn_stop = ttk.Button(top, text="감시 중단", command=self.stop_watch, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)

        self.status_led = tk.Label(top, text="●", font=("Arial", 18, "bold"), fg="red")
        self.status_text = ttk.Label(top, text="STOPPED", width=10)
        self.status_led.pack(side=tk.LEFT)
        self.status_text.pack(side=tk.LEFT, padx=(4, 10))

        self.beep_var = tk.BooleanVar(value=True)
        self.chk_beep = ttk.Checkbutton(top, text="Beep ON", variable=self.beep_var, command=self.apply_settings)
        self.chk_beep.pack(side=tk.LEFT, padx=10)

        ttk.Label(top, text="Extra shots(0~5):").pack(side=tk.LEFT, padx=(16, 5))
        self.extra_var = tk.IntVar(value=self.cfg.extra_shots)
        self.spin_extra = ttk.Spinbox(top, from_=0, to=5, textvariable=self.extra_var, width=3, command=self.apply_settings)
        self.spin_extra.pack(side=tk.LEFT)

        ttk.Label(top, text="interval(s):").pack(side=tk.LEFT, padx=(10, 5))
        self.extra_interval_var = tk.DoubleVar(value=self.cfg.extra_shot_interval_sec)
        self.spin_interval = ttk.Spinbox(
            top, from_=0.1, to=2.0, increment=0.1,
            textvariable=self.extra_interval_var, width=5, command=self.apply_settings
        )
        self.spin_interval.pack(side=tk.LEFT)

        # ✅ 미리보기 사진 번호 선택
        ttk.Label(top, text="미리보기(1~5):").pack(side=tk.LEFT, padx=(16, 5))
        self.preview_no_var = tk.IntVar(value=self.cfg.preview_shot_no)
        self.spin_preview_no = ttk.Spinbox(top, from_=1, to=5, textvariable=self.preview_no_var, width=3,
                                           command=self.apply_settings)
        self.spin_preview_no.pack(side=tk.LEFT)

        # ✅ 자동 따라가기 체크박스
        self.auto_follow_var = tk.BooleanVar(value=True)
        self.chk_autofollow = ttk.Checkbutton(top, text="자동 따라가기", variable=self.auto_follow_var)
        self.chk_autofollow.pack(side=tk.LEFT, padx=(14, 5))

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)

        self.btn_open_folder = ttk.Button(top, text="captures 폴더 열기", command=self.open_captures_folder)
        self.btn_clear_folder = ttk.Button(top, text="captures 사진 전부 삭제", command=self.clear_captures_folder)
        self.btn_open_folder.pack(side=tk.LEFT, padx=5)
        self.btn_clear_folder.pack(side=tk.LEFT, padx=5)

        # ===== SENSITIVITY ROW =====
        mid = ttk.Frame(self, padding=10)
        mid.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(mid, text="빛 예민강도 (1~10, 10이 가장 민감):").pack(side=tk.LEFT)
        self.sens_var = tk.IntVar(value=self.cfg.sensitivity_level)

        # ✅ 콜백 타이밍 문제 방지: command 등록을 나중에 한다
        self.scale = ttk.Scale(mid, from_=1, to=10, orient=tk.HORIZONTAL)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.scale.set(self.cfg.sensitivity_level)

        self.sens_label = ttk.Label(mid, text=f"{self.cfg.sensitivity_level}/10")
        self.sens_label.pack(side=tk.LEFT)

        self.scale.configure(command=self._on_scale)

        info = ttk.Frame(self, padding=(10, 0, 10, 10))
        info.pack(side=tk.TOP, fill=tk.X)
        self.params_label = ttk.Label(info, text="")
        self.params_label.pack(side=tk.LEFT)

        # ===== MAIN =====
        main = ttk.Frame(self, padding=10)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: log
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.txt = tk.Text(left, wrap=tk.NONE)
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.txt.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt.config(yscrollcommand=scroll_y.set)

        # Right: status board
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        board = ttk.LabelFrame(right, text="상태판", padding=10)
        board.pack(side=tk.TOP, fill=tk.X)

        self.lbl_last_detect = ttk.Label(board, text="마지막 감지: -", width=52)
        self.lbl_last_save = ttk.Label(board, text="마지막 저장: -", width=52)
        self.lbl_last_detect.pack(anchor="w", pady=2)
        self.lbl_last_save.pack(anchor="w", pady=2)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        gates = ttk.LabelFrame(right, text="입구별 상태", padding=10)
        gates.pack(side=tk.TOP, fill=tk.X)

        hdr = ttk.Frame(gates)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="입구", width=10).grid(row=0, column=0, sticky="w")
        ttk.Label(hdr, text="최근 ratio", width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(hdr, text="최근 감지시간", width=12).grid(row=0, column=2, sticky="w")
        ttk.Label(hdr, text="오늘 누적", width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(hdr, text="최근 저장파일", width=30).grid(row=0, column=4, sticky="w")

        self.gate_rows = {}
        for g in range(1, 5):
            row = ttk.Frame(gates)
            row.pack(fill=tk.X, pady=2)

            lbl_gate = ttk.Label(row, text=GATE_NAMES[g], width=10)
            lbl_ratio = ttk.Label(row, text="0.0000", width=12, style="Old.TLabel")
            lbl_time = ttk.Label(row, text="-", width=12, style="Old.TLabel")
            lbl_count = ttk.Label(row, text="0", width=10, style="Old.TLabel")
            lbl_file = ttk.Label(row, text="-", width=30, style="Old.TLabel")

            lbl_gate.grid(row=0, column=0, sticky="w")
            lbl_ratio.grid(row=0, column=1, sticky="w")
            lbl_time.grid(row=0, column=2, sticky="w")
            lbl_count.grid(row=0, column=3, sticky="w")
            lbl_file.grid(row=0, column=4, sticky="w")

            self.gate_rows[g] = (lbl_ratio, lbl_time, lbl_count, lbl_file)

        # ===== PREVIEW (우측하단) : 레이아웃 고정 =====
        preview = ttk.LabelFrame(right, text="미리보기", padding=10)
        preview.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))

        # ✅ "프레임 크기 고정" (내용에 의해 움직이지 않게)
        PREVIEW_W = 560
        PREVIEW_H = 330
        preview.configure(width=PREVIEW_W, height=PREVIEW_H)
        preview.pack_propagate(False)   # ✅ 핵심: 내용이 크기/위치를 흔들지 못하게

        # 미리보기 이미지 최대 크기(축소만)
        self.preview_max_w = 520
        self.preview_max_h = 260

        # ✅ 내부 배치를 grid로 고정
        preview.grid_propagate(False)

        # row0: 상단바 / row1: 이미지
        preview.grid_rowconfigure(0, weight=0)
        preview.grid_rowconfigure(1, weight=1)
        preview.grid_columnconfigure(0, weight=1)

        # --- 상단바 (고정 높이) ---
        topline = ttk.Frame(preview)
        topline.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        topline.grid_columnconfigure(0, weight=0)
        topline.grid_columnconfigure(1, weight=1)
        topline.grid_columnconfigure(2, weight=0)
        topline.grid_columnconfigure(3, weight=0)

        self.btn_prev_gate = ttk.Button(topline, text="◀", width=3, command=self.prev_preview_shot)
        self.btn_next_gate = ttk.Button(topline, text="▶", width=3, command=self.next_preview_shot)
        self.btn_latest = ttk.Button(topline, text="최근 샷", command=self.goto_latest_shot)
        self.btn_latest.grid(row=0, column=3, padx=(12, 0), sticky="e")


        # ✅ 게이트명 중앙 고정
        self.preview_gate_label = ttk.Label(topline, text=f"{GATE_NAMES[self.preview_gate]}", anchor="center")

        self.btn_prev_gate.grid(row=0, column=0, sticky="w")
        self.preview_gate_label.grid(row=0, column=1, sticky="ew", padx=10)
        self.btn_next_gate.grid(row=0, column=2, sticky="e")

        # --- 이미지 표시 영역 (고정) ---
        # ✅ Label 자체 크기를 고정해둬야 덜 흔들림
        self.preview_label = tk.Label(preview, bg="black", width=PREVIEW_W, height=PREVIEW_H)
        self.preview_label.grid(row=1, column=0, sticky="nsew")

        self.preview_imgtk = None

        # ===== BOTTOM =====
        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self.capture_count_label = ttk.Label(bottom, text="captures: 0", anchor="e")
        self.capture_count_label.pack(side=tk.RIGHT)

        self.apply_settings()
        self._update_capture_count()

    # -------------------------
    # Sens slider callback
    # -------------------------
    def _on_scale(self, _):
        lvl = int(round(float(self.scale.get())))
        self.sens_var.set(lvl)
        if hasattr(self, "sens_label"):
            self.sens_label.config(text=f"{lvl}/10")
        self.apply_settings()

    def _mapped_params_for_ui(self, lvl: int):
        lvl = max(1, min(10, int(lvl)))
        t = (lvl - 1) / 9.0
        motion_ratio = 0.030 + (0.008 - 0.030) * t
        thresh = int(round(35 + (20 - 35) * t))
        luma_jump = 5.0 + (15.0 - 5.0) * t
        edge_ratio_min = 0.004 + (0.001 - 0.004) * t
        return motion_ratio, thresh, luma_jump, edge_ratio_min

    # -------------------------
    # Settings apply + fallback
    # -------------------------
    def apply_settings(self):
        lvl = self._safe_int(self.sens_var.get(), default=self.cfg.sensitivity_level, min_v=1, max_v=10)
        extra = self._safe_int(self.extra_var.get(), default=self.cfg.extra_shots, min_v=0, max_v=5)
        interval = self._safe_float(
            self.extra_interval_var.get(),
            default=self.cfg.extra_shot_interval_sec,
            min_v=0.1, max_v=2.0
        )

        preview_no = self._safe_int(self.preview_no_var.get(), default=self.cfg.preview_shot_no, min_v=1, max_v=5)

        # ✅ 가능한 최대: 1 + extra_shots
        max_possible = 1 + extra
        if preview_no > max_possible:
            preview_no = max_possible
            self.preview_no_var.set(preview_no)

        with self.cfg_lock:
            self.cfg.beep_enabled = bool(self.beep_var.get())
            self.cfg.extra_shots = extra
            self.cfg.extra_shot_interval_sec = interval
            self.cfg.sensitivity_level = lvl
            self.cfg.preview_shot_no = preview_no

        motion_ratio, thresh, luma_jump, edge_ratio_min = self._mapped_params_for_ui(lvl)
        self.params_label.config(
            text=f"MR≈{motion_ratio:.3f}, TH={thresh}, LUMA_JUMP≈{luma_jump:.1f}, EDGE_MIN≈{edge_ratio_min:.4f}"
        )
        self.chk_beep.config(text=("Beep ON" if self.beep_var.get() else "Beep OFF"))

    # -------------------------
    # Start / Stop
    # -------------------------
    def start_watch(self):
        if not ROI_JSON.exists():
            messagebox.showerror("Error", f"rois.json not found:\n{ROI_JSON}")
            return

        self.apply_settings()

        self.watcher = WisenetWatcher(self.cfg, self.cfg_lock, self.logq, self.eventq)
        self.watcher.start()

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._log_ui("=== START ===")
        self._set_status_running(True)

    def stop_watch(self):
        if self.watcher is not None:
            self.watcher.stop()
            self.watcher = None

        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self._log_ui("=== STOP requested ===")
        self._set_status_running(False)

    def _set_status_running(self, running: bool):
        if running:
            self.status_led.config(fg="green")
            self.status_text.config(text="RUNNING")
        else:
            self.status_led.config(fg="red")
            self.status_text.config(text="STOPPED")

    # -------------------------
    # Captures folder utilities
    # -------------------------
    def open_captures_folder(self):
        try:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(str(SAVE_DIR))
        except Exception as e:
            messagebox.showerror("Error", f"폴더 열기 실패:\n{repr(e)}")

    def clear_captures_folder(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        count = self._count_capture_images()
        if count == 0:
            messagebox.showinfo("Info", "삭제할 이미지가 없습니다.")
            return

        ok = messagebox.askyesno(
            "확인",
            f"captures 폴더의 이미지 {count}개를 전부 삭제할까요?\n\n(되돌릴 수 없습니다)"
        )
        if not ok:
            return

        deleted = 0
        failed = 0
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]

        for pat in patterns:
            for p in SAVE_DIR.glob(pat):
                try:
                    p.unlink()
                    deleted += 1
                except Exception:
                    failed += 1

        self._log_ui(f"=== CLEAR captures === deleted={deleted}, failed={failed}")
        self._update_capture_count()

        # 기록도 초기화
        self.saved_by_gate_seq = {1: {}, 2: {}, 3: {}, 4: {}}

        for g in range(1, 5):
            self.last_saved_file[g] = "-"
            lbl_ratio, lbl_time, lbl_count, lbl_file = self.gate_rows[g]
            lbl_file.config(text="-")

        self.last_global_saved = "-"
        self.lbl_last_save.config(text="마지막 저장: -")

        # 미리보기도 비우기
        self.preview_label.config(image="")
        self.preview_imgtk = None

    def _count_capture_images(self) -> int:
        if not SAVE_DIR.exists():
            return 0
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
        total = 0
        for pat in patterns:
            total += sum(1 for _ in SAVE_DIR.glob(pat))
        return total

    def _update_capture_count(self):
        n = self._count_capture_images()
        self.capture_count_label.config(text=f"captures: {n}")

    def _update_capture_count_loop(self):
        self._update_capture_count()
        self.after(500, self._update_capture_count_loop)

    # -------------------------
    # Recent color + daily reset
    # -------------------------
    def _update_recent_colors_loop(self):
        today = datetime.now().strftime("%Y%m%d")
        if today != self.count_date:
            self.count_date = today
            for g in range(1, 5):
                self.detect_count[g] = 0
                lbl_ratio, lbl_time, lbl_count, lbl_file = self.gate_rows[g]
                lbl_count.config(text="0")

        now = time.time()
        for g in range(1, 5):
            recent = (now - self.last_detect_epoch[g]) <= self.RECENT_WINDOW_SEC
            style = "Recent.TLabel" if recent else "Old.TLabel"
            lbl_ratio, lbl_time, lbl_count, lbl_file = self.gate_rows[g]
            lbl_ratio.config(style=style)
            lbl_time.config(style=style)
            lbl_count.config(style=style)
            lbl_file.config(style=style)

        self.after(250, self._update_recent_colors_loop)

    # -------------------------
    # Preview helpers
    # -------------------------
    def update_preview_image(self, img_path: Path):
        try:
            if not img_path.exists():
                return

            img = Image.open(img_path)
            # ✅ 절대 늘리지 않음(축소만)
            img.thumbnail((self.preview_max_w, self.preview_max_h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(img)
            self.preview_imgtk = imgtk  # 참조 유지
            self.preview_label.config(image=imgtk)
        except Exception as e:
            self._log_ui(f"[preview] failed: {repr(e)}")

    def _max_possible_shot_no(self) -> int:
        """현재 설정 extra_shots 기준으로 가능한 최대 미리보기 번호"""
        with self.cfg_lock:
            extra = int(getattr(self.cfg, "extra_shots", 2))
        extra = max(0, min(5, extra))
        return 1 + extra  # 1=기본샷 포함

    def next_preview_shot(self):
        # 1) 같은 게이트에서 다음 샷으로
        max_no = self._max_possible_shot_no()
        cur_no = int(self.preview_no_var.get())
        nxt_no = cur_no + 1

        if nxt_no <= max_no:
            self.preview_no_var.set(nxt_no)
            self.apply_settings()
            self.refresh_preview_for_current_gate()
            return

        # 2) 없으면 다음 게이트로 넘어가고 샷=1로
        self.preview_gate = 1 if self.preview_gate >= 4 else self.preview_gate + 1
        shot_no = int(self.preview_no_var.get())
        self.preview_gate_label.config(text=f"{GATE_NAMES[self.preview_gate]}  (#{shot_no})")
        self.preview_no_var.set(1)
        self.apply_settings()
        self.refresh_preview_for_current_gate()

    def prev_preview_shot(self):
        # 1) 같은 게이트에서 이전 샷으로
        cur_no = int(self.preview_no_var.get())
        prv_no = cur_no - 1

        if prv_no >= 1:
            self.preview_no_var.set(prv_no)
            self.apply_settings()
            self.refresh_preview_for_current_gate()
            return

        # 2) 없으면 이전 게이트로 넘어가고 샷=마지막으로
        self.preview_gate = 4 if self.preview_gate <= 1 else self.preview_gate - 1
        self.preview_gate_label.config(text=f"{GATE_NAMES[self.preview_gate]}")

        max_no = self._max_possible_shot_no()
        self.preview_no_var.set(max_no)
        self.apply_settings()
        self.refresh_preview_for_current_gate()


    def refresh_preview_for_current_gate(self):
        # 설정된 preview_shot_no(1..5) -> wanted_seq(0..4)
        with self.cfg_lock:
            preview_shot_no = int(getattr(self.cfg, "preview_shot_no", 2))
            extra_shots = int(getattr(self.cfg, "extra_shots", 2))

        # ✅ 안전 fallback: 실제 존재 가능한 최대 번호
        max_possible = 1 + max(0, min(5, extra_shots))
        if preview_shot_no > max_possible:
            preview_shot_no = max_possible

        wanted_seq = max(0, min(4, preview_shot_no - 1))

        seq_map = self.saved_by_gate_seq.get(self.preview_gate, {})
        if not seq_map:
            self.preview_label.config(image="")
            self.preview_imgtk = None
            return

        candidates = sorted(seq_map.keys())
        if wanted_seq in seq_map:
            chosen_seq = wanted_seq
        else:
            below = [s for s in candidates if s <= wanted_seq]
            chosen_seq = max(below) if below else max(candidates)

        filename = seq_map.get(chosen_seq)
        if filename:
            self.update_preview_image(SAVE_DIR / filename)

    def goto_latest_shot(self):
        """
        가장 최근에 저장된 샷(gate, seq)으로 미리보기 즉시 점프
        """
        if not self.latest_saved_ref:
            messagebox.showinfo("Info", "아직 저장된 이미지가 없습니다.")
            return

        gate, seq, filename = self.latest_saved_ref

        # 1) 게이트 이동
        if gate in (1, 2, 3, 4):
            self.preview_gate = gate
            # 게이트 라벨 표시(샷 번호도 보여주고 싶으면 여기서 같이)
            self.preview_gate_label.config(text=f"{GATE_NAMES[self.preview_gate]}")

        # 2) 샷 번호 이동 (seq=0 -> 1번째, seq=1 -> 2번째 ...)
        target_no = seq + 1

        # ✅ 현재 extra_shots 범위 밖이면 자동 fallback
        max_no = self._max_possible_shot_no()
        if target_no > max_no:
            target_no = max_no

        if target_no < 1:
            target_no = 1

        self.preview_no_var.set(target_no)
        self.apply_settings()
        self.refresh_preview_for_current_gate()

    # -------------------------
    # Event polling
    # -------------------------
    def _poll_events(self):
        while True:
            try:
                ev = self.eventq.get_nowait()
            except queue.Empty:
                break

            etype = ev.get("type")

            if etype == "status":
                self._set_status_running(bool(ev.get("running")))

            elif etype == "detected":
                gate = int(ev.get("gate", 0))
                ratio = float(ev.get("ratio", 0.0))
                when = ev.get("when", "-")

                if gate in self.gate_rows:
                    self.last_ratio[gate] = ratio
                    self.last_detect_time[gate] = when
                    self.last_detect_epoch[gate] = time.time()

                    self.detect_count[gate] += 1

                    lbl_ratio, lbl_time, lbl_count, lbl_file = self.gate_rows[gate]
                    lbl_ratio.config(text=f"{ratio:.4f}")
                    lbl_time.config(text=when[-8:] if len(when) >= 8 else when)
                    lbl_count.config(text=str(self.detect_count[gate]))

                self.last_global_detected = f"{GATE_NAMES.get(gate, gate)} @ {when}"
                self.lbl_last_detect.config(text=f"마지막 감지: {self.last_global_detected}")

                # ✅ 자동 따라가기: 마지막 감지 게이트로 미리보기 전환
                if self.auto_follow_var.get() and gate in (1, 2, 3, 4):
                    self.preview_gate = gate
                    self.preview_gate_label.config(text=GATE_NAMES[self.preview_gate])
                    self.refresh_preview_for_current_gate()

            elif etype == "saved":
                gate = int(ev.get("gate", 0))
                filename = ev.get("filename", "-")
                when = ev.get("when", "-")
                seq = int(ev.get("seq", 0))

                if gate in self.gate_rows:
                    self.last_saved_file[gate] = filename
                    lbl_ratio, lbl_time, lbl_count, lbl_file = self.gate_rows[gate]
                    lbl_file.config(text=filename[:28] + ("…" if len(filename) > 28 else ""))

                self.last_global_saved = f"{GATE_NAMES.get(gate, gate)} -> {filename}"
                self.lbl_last_save.config(text=f"마지막 저장: {self.last_global_saved}")
                # ✅ 가장 최근 저장 샷 갱신
                self.latest_saved_ref = (gate, seq, filename)

                # ✅ gate/seq별 파일 기록
                if gate in self.saved_by_gate_seq and isinstance(filename, str) and filename != "-":
                    self.saved_by_gate_seq[gate][seq] = filename

                # ✅ 미리보기 게이트가 현재 gate일 때만 즉시 갱신
                # (자동 따라가기 OFF라도 내 선택 고정되게)
                if gate == self.preview_gate:
                    self.refresh_preview_for_current_gate()

        self.after(100, self._poll_events)

    # -------------------------
    # Logging
    # -------------------------
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

    # -------------------------
    # Close
    # -------------------------
    def on_close(self):
        try:
            if self.watcher is not None:
                self.watcher.stop()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
