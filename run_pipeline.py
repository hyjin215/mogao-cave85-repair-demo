"""Mogao Cave 85 human-in-the-loop virtual restoration pipeline.
Run from any working directory: python run_pipeline.py
"""
from pathlib import Path
import json
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
INPUTS_DIR = BASE / "inputs"     # 新建inputs文件夹放多张原图
OUTPUT_ROOT = BASE / "outputs"
CHECKPOINT = BASE / "checkpoints" / "unetplusplus_best.pth"
THRESHOLD = 0.50  # 可改为 0.35、0.40、0.45 等；必须人工复核
WINDOW = "85 Cave | Left: add damage | Right: erase | S: finish | R: reset | ESC: quit"
MAX_W, MAX_H = 1400, 900


def read_image(path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入图片：{path}\n请将壁画图片放到 inputs 文件夹内。")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片：{path}")
    return img


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"保存失败：{path}")
    encoded.tofile(str(path))


def edit_mask(image, initial_mask):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8) if initial_mask is None else initial_mask.copy()
    if mask.shape != (h, w):
        raise ValueError("Mask 尺寸与原图不一致")
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
    sx, sy = w / dw, h / dh
    radius = max(2, int(min(w, h) * 0.006))
    last = None

    def mouse(event, x, y, flags, _):
        nonlocal last
        px = int(np.clip(round(x * sx), 0, w - 1))
        py = int(np.clip(round(y * sy), 0, h - 1))
        p = (px, py)
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(mask, p, radius, 255, -1); last = p
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(mask, p, radius, 0, -1); last = p
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                if last is not None: cv2.line(mask, last, p, 255, radius * 2)
                cv2.circle(mask, p, radius, 255, -1); last = p
            elif flags & cv2.EVENT_FLAG_RBUTTON:
                if last is not None: cv2.line(mask, last, p, 0, radius * 2)
                cv2.circle(mask, p, radius, 0, -1); last = p
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            last = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, dw, dh)
    cv2.setMouseCallback(WINDOW, mouse)
    print("\nMask 人工修正：左键=增加待修复区；右键=擦除；+/-=画笔大小；R=清空；S=结束编辑；ESC=退出")
    while True:
        show = image.copy()
        sel = mask > 0
        if np.any(sel):
            show[sel] = (show[sel].astype(np.float32) * 0.45 + np.array([0, 0, 255]) * 0.55).astype(np.uint8)
        show = cv2.resize(show, (dw, dh), interpolation=cv2.INTER_AREA)
        cv2.rectangle(show, (0, 0), (dw, 34), (30, 30, 30), -1)
        cv2.putText(show, f"Brush {radius}px | Left add | Right erase | S finish | R clear | ESC quit", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, show)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows(); return None
        if key in (ord('s'), ord('S')):
            cv2.destroyAllWindows(); return mask
        if key in (ord('r'), ord('R')):
            mask[:] = 0; print("Mask 已清空")
        if key in (ord('+'), ord('=')): radius = min(radius + 2, max(5, min(w,h)//10))
        if key in (ord('-'), ord('_')): radius = max(1, radius - 2)


def run_opencv(image, mask):
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def run_lama(image, mask):
    from PIL import Image
    from simple_lama_inpainting import SimpleLama
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = SimpleLama()(Image.fromarray(rgb), Image.fromarray(mask, mode="L"))
    result = np.asarray(result.convert("RGB"))
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # 保护未标记区域：只把模型生成结果写回 Mask 内
    merged = image.copy()
    merged[mask > 0] = result[mask > 0]
    return merged


def comparison(image, mask, cv_result, lama_result=None):
    panels = [("Original", image), ("Expert-confirmed Mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), ("OpenCV Telea", cv_result)]
    if lama_result is not None: panels.append(("LaMa", lama_result))
    thumbs = []
    for title, panel in panels:
        h, w = panel.shape[:2]
        scale = min(500 / w, 500 / h, 1.0)
        thumb = cv2.resize(panel, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        canvas = np.full((thumb.shape[0]+36, thumb.shape[1], 3), 245, np.uint8)
        canvas[36:] = thumb
        cv2.putText(canvas, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (20,20,20), 2, cv2.LINE_AA)
        thumbs.append(canvas)
    return cv2.hconcat(thumbs)


def pick_image_from_folder():
    INPUTS_DIR.mkdir(exist_ok=True)
    exts = {".jpg",".jpeg",".png",".bmp"}
    files = [p for p in sorted(INPUTS_DIR.glob("*")) if p.suffix.lower() in exts and p.is_file()]
    if len(files) == 0:
        raise FileNotFoundError(f"在 {INPUTS_DIR} 没有找到图片，请把壁画图片放到inputs文件夹。")
    print("\n==== inputs 文件夹图片列表 ====")
    for idx, f in enumerate(files):
        print(f"[{idx}] {f.name}")
    while True:
        sel = input("\n请输入要处理图片的序号：").strip()
        if sel.isdigit():
            n = int(sel)
            if 0 <= n < len(files):
                return files[n]
        print("输入无效，请输入列表里的数字序号。")


def main():
    print("=== 第85窟：AI识别—人工修正—审核确认—虚拟修复 ===")
    img_path = pick_image_from_folder()
    image = read_image(img_path)
    h, w = image.shape[:2]
    OUTPUT_ROOT.mkdir(exist_ok=True)
    print(f"输入图片：{img_path.name}，尺寸 {w}x{h}")

    initial_mask = None
    prob = None
    if CHECKPOINT.is_file():
        try:
            from damage_segmentation import predict_damage_mask
            print("加载 U-Net++ 权重并预测疑似破损区域……")
            initial_mask, prob = predict_damage_mask(image, CHECKPOINT, threshold=THRESHOLD)
            save_image(OUTPUT_ROOT / "mask_auto.png", initial_mask)
            save_image(OUTPUT_ROOT / "mask_probability.png", prob)
            print("自动预测完成。红色区域只是 AI 候选结果，不代表最终病害结论。")
        except Exception as e:
            print(f"自动识别未能运行：{e}\n转为从空白 Mask 手工标记。")
    else:
        print(f"没有找到权重：{CHECKPOINT}\n本次从空白 Mask 开始手工标记。")

    mask = edit_mask(image, initial_mask)
    if mask is None:
        print("用户退出，未执行修复。"); return
    if not np.any(mask):
        print("Mask 为空，没有待修复区域，已停止。"); return

    save_image(OUTPUT_ROOT / "mask_pending_review.png", mask)
    print(f"当前待修复像素比例：{np.mean(mask > 0)*100:.2f}%")
    print("请检查窗口中的红色区域，并由负责人员确认修复范围。")
    answer = input("确认该 Mask 已经人工检查/专家确认？输入 YES 才继续：").strip()
    if answer != "YES":
        print("未确认，程序停止。Mask 已保存到 outputs/mask_pending_review.png；不会执行修复。")
        return

    save_image(OUTPUT_ROOT / "mask_expert_confirmed.png", mask)
    save_image(OUTPUT_ROOT / "mask.png", mask)
    save_image(OUTPUT_ROOT / "original.png", image)
    print("运行 OpenCV Telea……")
    cv_result = run_opencv(image, mask)
    save_image(OUTPUT_ROOT / "result_opencv.png", cv_result)

    lama_result = None
    try:
        print("运行 LaMa（首次运行可能下载模型）……")
        lama_result = run_lama(image, mask)
        save_image(OUTPUT_ROOT / "result_lama.png", lama_result)
    except Exception as e:
        print(f"LaMa 本次未完成：{e}\nOpenCV 结果仍已保存。")

    comp = comparison(image, mask, cv_result, lama_result)
    save_image(OUTPUT_ROOT / "Final_Comparison.jpg", comp)
    metadata = {
        "input_image": img_path.name,
        "image_width": w, "image_height": h,
        "checkpoint": str(CHECKPOINT) if CHECKPOINT.exists() else None,
        "ai_threshold": THRESHOLD if CHECKPOINT.exists() else None,
        "mask_pixels_percent": round(float(np.mean(mask > 0) * 100), 4),
        "human_review_confirmed": True,
        "note": "AI output is a virtual restoration proposal, not a physical conservation instruction."
    }
    (OUTPUT_ROOT / "review_record.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n完成。输出文件位于：", OUTPUT_ROOT)
    for p in sorted(OUTPUT_ROOT.iterdir()):
        if p.is_file(): print(" -", p.name)
    view = comp.copy()
    if view.shape[1] > 1600:
        ratio = 1600 / view.shape[1]
        view = cv2.resize(view, (1600, int(view.shape[0]*ratio)))
    cv2.imshow("85 Cave Restoration Comparison - press any key", view)
    cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n已中断")
    except Exception as exc:
        print(f"程序错误：{exc}")
        raise
