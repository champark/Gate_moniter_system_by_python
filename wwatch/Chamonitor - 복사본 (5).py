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
CHECK_JSON = BASE_DIR / "checked.json"   # ✅ 수동 확인 상태 저장 파일

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
    sensitivity_level: int = 2     # 1..10 (10 most sensitive)
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

    preview_shot_no: int = 2

    font_path: str = r"C:\Windows\Fonts\malgun.ttf"
    font_size: int = 24


class ChamWatcher(threading.Thread):
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

        txt = f"{gate} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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

        self.log("Watching... (use Stop button)")
        self.emit({"type": "status", "running": True})

        with mss.mss() as sct:
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

                    # ---- pending extra shots ----
                    if pending[i]:
                        ready = [x for x in pending[i] if x[0] <= now]
                        pending[i] = [x for x in pending[i] if x[0] > now]
                        for _, event_ts, seq in ready:
                            try:
                                shot2 = np.array(sct.grab(roi))
                                bgr2 = cv2.cvtColor(shot2, cv2.COLOR_BGRA2BGR)
                                saved2 = self.save_image(view_idx, bgr2, 0.0, event_ts, seq, font_path, font_size)
                                self.emit({
                                    "type": "saved",
                                    "gate": view_idx,
                                    "filename": saved2.name,
                                    "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "seq": seq,
                                })
                            except Exception as e:
                                self.log(f"  FAILED extra save: {repr(e)}")

                    # ---- capture ----
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
                        self.log(f"[{ts_human}] 감지: 입구{view_idx}")

                        self.emit({
                            "type": "detected",
                            "gate": view_idx,
                            "ratio": ratio,
                            "when": ts_human,
                        })

                        if beep_enabled:
                            do_beep()

                        if now - last_saved_at[i] >= cooldown:
                            event_ts = safe_timestamp()
                            try:
                                saved = self.save_image(view_idx, bgr, ratio, event_ts, seq=0,
                                                        font_path=font_path, font_size=font_size)
                                last_saved_at[i] = now
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
    RECENT_WINDOW_SEC = 8.0

    def __init__(self):
        super().__init__()
        self.title("Cham Is Watching YOU")
        self.geometry("800x564")

        self.cfg = RuntimeConfig()
        self.cfg_lock = threading.Lock()
        self.logq = queue.Queue()
        self.eventq = queue.Queue()
        self.watcher = None

        # ✅ 체크 상태 로드(프로그램 재실행해도 유지)
        self.checked = set()
        self._load_checked()

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

        # ✅ 미리보기 리사이즈용
        self.preview_imgtk = None
        self.current_preview_path: Path | None = None
        self._preview_resize_job = None

        # ✅ 확인 체크 UI 제어용
        self._check_var = tk.BooleanVar(value=False)
        self._check_updating = False

        self._build_styles()
        self._build_ui()

        # ✅ (중요) 실행 전 저장돼 있던 캡쳐들도 로딩
        self._load_existing_captures_to_memory()
        self._goto_latest_if_exists()  # 있으면 바로 최신으로 보여줌
        self._update_capture_count()

        self.after(100, self._poll_logs)
        self.after(100, self._poll_events)
        self.after(500, self._update_capture_count_loop)
        self.after(250, self._update_recent_colors_loop)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------
    # checked persistence
    # -------------------------
    def _load_checked(self):
        try:
            if CHECK_JSON.exists():
                data = json.loads(CHECK_JSON.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.checked = set(data)
        except Exception:
            self.checked = set()

    def _save_checked(self):
        try:
            CHECK_JSON.write_text(
                json.dumps(sorted(self.checked), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    # -------------------------
    # ✅ 기존 캡쳐 파일도 로딩
    # -------------------------
    def _load_existing_captures_to_memory(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]

        files = []
        for pat in patterns:
            files.extend(list(SAVE_DIR.glob(pat)))

        # reset
        self.saved_by_gate_seq = {1: {}, 2: {}, 3: {}, 4: {}}

        for p in files:
            fn = p.name
            gate = self._infer_gate_from_filename(fn)
            seq = self._infer_seq_from_filename(fn)
            if gate in (1, 2, 3, 4):
                # 같은 gate/seq에 중복이 있으면 더 최신(파일명 큰것)으로
                old = self.saved_by_gate_seq[gate].get(seq)
                if (old is None) or (fn > old):
                    self.saved_by_gate_seq[gate][seq] = fn

        # latest ref도 갱신
        items = self._list_all_saved_files()
        if items:
            fn, gate, seq = items[-1]
            self.latest_saved_ref = (gate, seq, fn)

    def _infer_gate_from_filename(self, filename: str) -> int | None:
        for g, name in GATE_NAMES.items():
            if name in filename:
                return g
        return None

    def _infer_seq_from_filename(self, filename: str) -> int:
        try:
            stem = Path(filename).stem
            parts = stem.split("_")
            last = parts[-1]
            if last.isdigit():
                return int(last)
        except Exception:
            pass
        return 0

    def _goto_latest_if_exists(self):
        items = self._list_all_saved_files()
        if not items:
            self._sync_checked_ui()
            return
        fn, gate, seq = items[-1]
        self._jump_to_global_item(fn, gate, seq)

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

        ttk.Label(top, text="미리보기(1~5):").pack(side=tk.LEFT, padx=(16, 5))
        self.preview_no_var = tk.IntVar(value=self.cfg.preview_shot_no)
        self.spin_preview_no = ttk.Spinbox(top, from_=1, to=5, textvariable=self.preview_no_var, width=3,
                                           command=self.apply_settings)
        self.spin_preview_no.pack(side=tk.LEFT)

        self.auto_follow_var = tk.BooleanVar(value=True)
        self.chk_autofollow = ttk.Checkbutton(top, text="자동 따라가기", variable=self.auto_follow_var)
        self.chk_autofollow.pack(side=tk.LEFT, padx=(14, 5))

        # ===== SENSITIVITY ROW =====
        mid = ttk.Frame(self, padding=10)
        mid.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(mid, text="빛 예민강도 (1~10, 10이 가장 민감):").pack(side=tk.LEFT)
        self.sens_var = tk.IntVar(value=self.cfg.sensitivity_level)

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

        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_rowconfigure(0, weight=0)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # ===== LEFT TOP: LOG =====
        log_frame = ttk.LabelFrame(left, text="로그", padding=6)
        log_frame.grid(row=0, column=0, sticky="ew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)

        self.txt = tk.Text(log_frame, wrap=tk.NONE, height=7)
        self.txt.grid(row=0, column=0, sticky="ew")

        scroll_y = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.txt.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")
        self.txt.config(yscrollcommand=scroll_y.set)

        # ===== LEFT BOTTOM: STATUS =====
        status_frame = ttk.Frame(left)
        status_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_rowconfigure(0, weight=0)
        status_frame.grid_rowconfigure(1, weight=1)

        board = ttk.LabelFrame(status_frame, text="상태판", padding=10)
        board.grid(row=0, column=0, sticky="ew")

        self.lbl_last_detect = ttk.Label(board, text="마지막 감지: -", width=70)
        self.lbl_last_save = ttk.Label(board, text="마지막 저장: -", width=70)
        self.lbl_last_detect.pack(anchor="w", pady=2)
        self.lbl_last_save.pack(anchor="w", pady=2)

        gates = ttk.LabelFrame(status_frame, text="입구별 상태", padding=10)
        gates.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        gates.grid_columnconfigure(0, weight=1)

        hdr = ttk.Frame(gates)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="입구", width=10).grid(row=0, column=0, sticky="w")
        ttk.Label(hdr, text="최근 감지시간", width=12).grid(row=0, column=2, sticky="w")
        ttk.Label(hdr, text="누적", width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(hdr, text="최근 저장파일", width=36).grid(row=0, column=4, sticky="w")

        self.gate_rows = {}
        for g in range(1, 5):
            row = ttk.Frame(gates)
            row.pack(fill=tk.X, pady=2)

            lbl_gate = ttk.Label(row, text=GATE_NAMES[g], width=10)
            lbl_time = ttk.Label(row, text="-", width=12, style="Old.TLabel")
            lbl_count = ttk.Label(row, text="0", width=10, style="Old.TLabel")
            lbl_file = ttk.Label(row, text="-", width=36, style="Old.TLabel")

            lbl_gate.grid(row=0, column=0, sticky="w")
            lbl_time.grid(row=0, column=2, sticky="w")
            lbl_count.grid(row=0, column=3, sticky="w")
            lbl_file.grid(row=0, column=4, sticky="w")

            self.gate_rows[g] = (lbl_time, lbl_count, lbl_file)

        # ===== RIGHT: PREVIEW =====
        preview = ttk.LabelFrame(right, text="미리보기", padding=10)
        preview.configure(width=520, height=360)
        preview.grid_propagate(False)
        preview.grid(row=0, column=0, sticky="nsew")

        preview.pack_propagate(False)
        preview.grid_propagate(False)

        preview.grid_rowconfigure(0, weight=0)
        preview.grid_rowconfigure(1, weight=1)
        preview.grid_rowconfigure(2, weight=0)
        preview.grid_columnconfigure(0, weight=1)

        topline = ttk.Frame(preview)
        topline.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        topline.grid_columnconfigure(0, weight=0)
        topline.grid_columnconfigure(1, weight=1)
        topline.grid_columnconfigure(2, weight=0)
        topline.grid_columnconfigure(3, weight=0)

        self.btn_prev_gate = ttk.Button(topline, text="이전 샷", width=7, command=self.prev_preview_shot)
        self.btn_next_gate = ttk.Button(topline, text="다음 샷", width=7, command=self.next_preview_shot)
        self.btn_latest = ttk.Button(topline, text="최근 샷", command=self.goto_latest_shot)

        self.preview_gate_label = ttk.Label(topline, text=f"{GATE_NAMES[self.preview_gate]}", anchor="center")

        self.btn_prev_gate.grid(row=0, column=0, sticky="w")
        self.preview_gate_label.grid(row=0, column=1, sticky="ew", padx=10)
        self.btn_next_gate.grid(row=0, column=2, sticky="e")
        self.btn_latest.grid(row=0, column=3, padx=(12, 0), sticky="e")

        self.preview_label = tk.Label(preview, bg="black")
        self.preview_label.grid(row=1, column=0, sticky="nsew")
        self.preview_label.bind("<Configure>", self._on_preview_configure)

        # ✅ 체크 + 상태 아이콘
        checkline = ttk.Frame(preview)
        checkline.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        checkline.grid_columnconfigure(0, weight=1)
        checkline.grid_columnconfigure(1, weight=0)

        # 왼쪽 상태표시 (✅ 확인됨 / ❌ 미확인)
        self.lbl_checked_mark = ttk.Label(checkline, text="", font=("Arial", 11, "bold"))
        self.lbl_checked_mark.grid(row=0, column=0, sticky="w")

        self.chk_confirm = ttk.Checkbutton(
            checkline, text="확인됨", variable=self._check_var,
            command=self._on_toggle_checked
        )
        self.chk_confirm.grid(row=0, column=1, sticky="e")

        # ===== BOTTOM =====
        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        # ✅ 총 캡쳐 수 + 미확인 수
        self.capture_count_label = ttk.Label(bottom, text="총 캡쳐 수 : 0 / 미확인 : 0", anchor="e")
        self.capture_count_label.pack(side=tk.RIGHT)

        self.btn_open_folder = ttk.Button(bottom, text="captures 폴더 열기", command=self.open_captures_folder)
        self.btn_clear_folder = ttk.Button(bottom, text="captures 사진 전부 삭제", command=self.clear_captures_folder)
        self.btn_open_folder.pack(side=tk.RIGHT, padx=5)
        self.btn_clear_folder.pack(side=tk.RIGHT, padx=5)

        self.apply_settings()

    # -------------------------
    # Preview resize (debounced)
    # -------------------------
    def _on_preview_configure(self, _evt=None):
        if self._preview_resize_job is not None:
            try:
                self.after_cancel(self._preview_resize_job)
            except Exception:
                pass
        self._preview_resize_job = self.after(120, self._redraw_preview_current)

    def _redraw_preview_current(self):
        self._preview_resize_job = None
        if self.current_preview_path and self.current_preview_path.exists():
            self._render_preview_image(self.current_preview_path)

    def _render_preview_image(self, img_path: Path):
        try:
            if not img_path.exists():
                return

            w = max(50, int(self.preview_label.winfo_width()) - 10)
            h = max(50, int(self.preview_label.winfo_height()) - 10)

            img = Image.open(img_path)
            img.thumbnail((w, h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(img)
            self.preview_imgtk = imgtk
            self.preview_label.config(image=imgtk)

        except Exception as e:
            self._log_ui(f"[preview] failed: {repr(e)}")

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
    # Settings apply
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

        max_possible = 1 + extra
        if preview_no > max_possible:
            preview_no = 1
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

        self.watcher = ChamWatcher(self.cfg, self.cfg_lock, self.logq, self.eventq)
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

        # 메모리 상태 초기화
        self.saved_by_gate_seq = {1: {}, 2: {}, 3: {}, 4: {}}
        self.latest_saved_ref = None
        self.current_preview_path = None
        self.preview_label.config(image="")
        self.preview_imgtk = None

        # 체크도 초기화
        self.checked = set()
        self._save_checked()

        for g in range(1, 5):
            self.last_saved_file[g] = "-"
            lbl_time, lbl_count, lbl_file = self.gate_rows[g]
            lbl_file.config(text="-")

        self.last_global_saved = "-"
        self.lbl_last_save.config(text="마지막 저장: -")

        self._sync_checked_ui()
        self._update_capture_count()

    # -------------------------
    # counts
    # -------------------------
    def _count_capture_images(self) -> int:
        if not SAVE_DIR.exists():
            return 0
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
        total = 0
        for pat in patterns:
            total += sum(1 for _ in SAVE_DIR.glob(pat))
        return total

    def _list_existing_image_names(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
        names = []
        for pat in patterns:
            for p in SAVE_DIR.glob(pat):
                names.append(p.name)
        return names

    def _update_capture_count(self):
        names = self._list_existing_image_names()
        total = len(names)

        # ✅ 실제로 존재하는 것 중에서만 미확인 계산
        unchecked = sum(1 for n in names if n not in self.checked)

        self.capture_count_label.config(text=f"총 캡쳐 수 : {total} / 미확인 : {unchecked}")

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
                lbl_time, lbl_count, lbl_file = self.gate_rows[g]
                lbl_count.config(text="0")

        now = time.time()
        for g in range(1, 5):
            recent = (now - self.last_detect_epoch[g]) <= self.RECENT_WINDOW_SEC
            style = "Recent.TLabel" if recent else "Old.TLabel"
            lbl_time, lbl_count, lbl_file = self.gate_rows[g]
            lbl_time.config(style=style)
            lbl_count.config(style=style)
            lbl_file.config(style=style)

        self.after(250, self._update_recent_colors_loop)

    # -------------------------
    # Global list navigation
    # -------------------------
    def _list_all_saved_files(self):
        items = []
        for gate in (1, 2, 3, 4):
            seq_map = self.saved_by_gate_seq.get(gate, {})
            for seq, fn in seq_map.items():
                if isinstance(fn, str) and fn:
                    items.append((fn, gate, seq))
        items.sort(key=lambda x: x[0])
        return items

    def _get_current_index_in_global(self):
        if not self.current_preview_path:
            return None, None
        cur_name = self.current_preview_path.name
        items = self._list_all_saved_files()
        for idx, (fn, gate, seq) in enumerate(items):
            if fn == cur_name:
                return idx, items
        return None, items

    def _jump_to_global_item(self, fn: str, gate: int, seq: int):
        path = SAVE_DIR / fn
        if not path.exists():
            return
        self.preview_gate = gate
        self.preview_gate_label.config(text=f"{GATE_NAMES.get(gate, gate)}")
        self.current_preview_path = path
        self._render_preview_image(path)

        self.latest_saved_ref = (gate, seq, fn)
        self._sync_checked_ui()
        self._update_capture_count()

    def next_preview_shot(self):
        idx, items = self._get_current_index_in_global()
        if not items:
            return

        if idx is None:
            fn, gate, seq = items[-1]
            self._jump_to_global_item(fn, gate, seq)
            return

        if idx >= len(items) - 1:
            return  # 최신이면 이동 금지

        fn, gate, seq = items[idx + 1]
        self._jump_to_global_item(fn, gate, seq)

    def prev_preview_shot(self):
        idx, items = self._get_current_index_in_global()
        if not items:
            return

        if idx is None:
            fn, gate, seq = items[0]
            self._jump_to_global_item(fn, gate, seq)
            return

        if idx <= 0:
            return

        fn, gate, seq = items[idx - 1]
        self._jump_to_global_item(fn, gate, seq)

    def goto_latest_shot(self):
        items = self._list_all_saved_files()
        if not items:
            messagebox.showinfo("Info", "아직 저장된 이미지가 없습니다.")
            return
        fn, gate, seq = items[-1]
        self._jump_to_global_item(fn, gate, seq)

    # -------------------------
    # Manual checked toggle + UI sync
    # -------------------------
    def _sync_checked_ui(self):
        self._check_updating = True
        try:
            if not self.current_preview_path:
                self._check_var.set(False)
                self.lbl_checked_mark.config(text="", foreground="gray")
                return

            name = self.current_preview_path.name
            is_checked = (name in self.checked)
            self._check_var.set(is_checked)

            if is_checked:
                self.lbl_checked_mark.config(text="✅ 확인됨", foreground="green")
            else:
                self.lbl_checked_mark.config(text="❌ 미확인", foreground="red")
        finally:
            self._check_updating = False

    def _on_toggle_checked(self):
        if self._check_updating:
            return
        if not self.current_preview_path:
            self._check_updating = True
            self._check_var.set(False)
            self._check_updating = False
            return

        name = self.current_preview_path.name
        want = bool(self._check_var.get())

        if want:
            self.checked.add(name)
        else:
            if name in self.checked:
                self.checked.remove(name)

        self._save_checked()
        self._sync_checked_ui()
        self._update_capture_count()

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
                when = ev.get("when", "-")

                if gate in self.gate_rows:
                    self.last_detect_time[gate] = when
                    self.last_detect_epoch[gate] = time.time()
                    self.detect_count[gate] += 1

                    lbl_time, lbl_count, lbl_file = self.gate_rows[gate]
                    lbl_time.config(text=when[-8:] if len(when) >= 8 else when)
                    lbl_count.config(text=str(self.detect_count[gate]))

                self.last_global_detected = f"{GATE_NAMES.get(gate, gate)} @ {when}"
                self.lbl_last_detect.config(text=f"마지막 감지: {self.last_global_detected}")

                if self.auto_follow_var.get() and gate in (1, 2, 3, 4):
                    self.preview_gate = gate
                    self.preview_gate_label.config(text=GATE_NAMES[self.preview_gate])

            elif etype == "saved":
                gate = int(ev.get("gate", 0))
                filename = ev.get("filename", "-")
                when = ev.get("when", "-")
                seq = int(ev.get("seq", 0))

                if gate in self.gate_rows:
                    self.last_saved_file[gate] = filename
                    lbl_time, lbl_count, lbl_file = self.gate_rows[gate]
                    lbl_file.config(text=filename[:34] + ("…" if len(filename) > 34 else ""))

                self.last_global_saved = f"{GATE_NAMES.get(gate, gate)} -> {filename}"
                self.lbl_last_save.config(text=f"마지막 저장: {self.last_global_saved}")

                # ✅ 메모리 기록
                if gate in self.saved_by_gate_seq and isinstance(filename, str) and filename != "-":
                    self.saved_by_gate_seq[gate][seq] = filename

                # ✅ latest 갱신
                self.latest_saved_ref = (gate, seq, filename)

                # ✅ 자동 따라가기면 화면 갱신
                if gate == self.preview_gate:
                    self.current_preview_path = SAVE_DIR / filename
                    self._render_preview_image(self.current_preview_path)
                    self._sync_checked_ui()
                    self._update_capture_count()

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
