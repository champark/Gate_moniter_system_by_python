import json
from pathlib import Path

import mss
import numpy as np
import cv2

N = 4
MAX_W, MAX_H = 1600, 900  # 표시용(필요시 조절)

def main():
    out_json = Path(__file__).resolve().parent / "rois.json"
    print("Will save to:", out_json)

    with mss.mss() as sct:
        print("\nAvailable monitors:")
        for i, m in enumerate(sct.monitors):
            print(i, m)
        print("Note: index 0 is 'all monitors'. Use 1..N for specific monitor.\n")

        idx = int(input("Which monitor index to calibrate? (e.g., 1 or 2): ").strip())
        if idx <= 0 or idx >= len(sct.monitors):
            raise ValueError("Pick a monitor index from 1..N shown above.")

        mon = sct.monitors[idx]
        # 이 캡처는 모니터 내부 이미지(좌표 원점은 모니터의 left/top)
        img0 = np.array(sct.grab(mon))  # BGRA
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)

    h0, w0 = img0.shape[:2]

    # 표시용 축소 비율(좌표는 나중에 역변환)
    scale = min(MAX_W / w0, MAX_H / h0, 1.0)
    disp = cv2.resize(img0, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img0.copy()

    rois = []
    win = "ROI selector (Drag => Enter confirm, R redo, ESC cancel)"
    dragging = False
    x0 = y0 = x1 = y1 = 0
    has_box = False

    def redraw():
        canvas = disp.copy()
        cv2.putText(canvas, f"Select ROI {len(rois)+1}/{N} (Monitor {idx})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        if has_box:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow(win, canvas)

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, x0, y0, x1, y1, has_box
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            x0, y0 = x, y
            x1, y1 = x, y
            has_box = True
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            x1, y1 = x, y
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            x1, y1 = x, y
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("Canceled.")
            break

        if key in (ord('r'), ord('R')):
            has_box = False
            redraw()
            continue

        if key in (13, 10):  # Enter
            if not has_box:
                continue

            xa, xb = sorted([x0, x1])
            ya, yb = sorted([y0, y1])
            w = xb - xa
            h = yb - ya
            if w < 5 or h < 5:
                print("Too small. Drag a bigger rectangle.")
                has_box = False
                redraw()
                continue

            inv = 1.0 / scale

            # ★핵심: 모니터 좌표 -> 전체 가상화면 절대좌표로 변환
            left = mon["left"] + int(round(xa * inv))
            top  = mon["top"]  + int(round(ya * inv))
            width  = int(round(w * inv))
            height = int(round(h * inv))

            rois.append({"left": left, "top": top, "width": width, "height": height})
            print(f"ROI {len(rois)} saved:", rois[-1])

            has_box = False
            redraw()

            if len(rois) == N:
                out_json.write_text(json.dumps(rois, ensure_ascii=False, indent=2), encoding="utf-8")
                print("Saved:", out_json)
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
