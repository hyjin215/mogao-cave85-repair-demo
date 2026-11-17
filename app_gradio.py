import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from damage_segmentation import predict_damage_mask

BASE = Path(__file__).resolve().parent
CHECKPOINT = BASE / "checkpoints" / "unetplusplus_best.pth"


def run_opencv(image, mask):
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def run_lama(image, mask):
    from PIL import Image
    from simple_lama_inpainting import SimpleLama
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = SimpleLama()(Image.fromarray(rgb), Image.fromarray(mask, mode="L"))
    result = np.asarray(result.convert("RGB"))
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    merged = image.copy()
    merged[mask > 0] = result[mask > 0]
    return merged


def pipeline(img, threshold=0.5):
    # 转成BGR，和原管线保持一致
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if CHECKPOINT.is_file():
        initial_mask, prob = predict_damage_mask(image, CHECKPOINT, threshold=threshold)
    else:
        h, w = image.shape[:2]
        initial_mask = np.zeros((h, w), dtype=np.uint8)

    cv_result = run_opencv(image, initial_mask)
    try:
        lama_result = run_lama(image, initial_mask)
    except Exception as e:
        lama_result = None

    # 拼接对比图
    panels = [
        ("Original", image),
        ("Auto Mask", cv2.cvtColor(initial_mask, cv2.COLOR_GRAY2BGR)),
        ("OpenCV Telea", cv_result)
    ]
    if lama_result is not None:
        panels.append(("LaMa", lama_result))

    thumbs = []
    for title, panel in panels:
        h, w = panel.shape[:2]
        scale = min(500 / w, 500 / h, 1.0)
        thumb = cv2.resize(panel, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        canvas = np.full((thumb.shape[0]+36, thumb.shape[1], 3), 245, np.uint8)
        canvas[36:] = thumb
        cv2.putText(canvas, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (20,20,20), 2, cv2.LINE_AA)
        thumbs.append(canvas)
    concat = cv2.hconcat(thumbs)
    return cv2.cvtColor(concat, cv2.COLOR_BGR2RGB)


with gr.Blocks(title="莫高窟第85窟虚拟修复") as demo:
    gr.Markdown("# 莫高窟第85窟 AI破损识别 + 虚拟修复")
    gr.Markdown("上传壁画原图，自动生成破损mask，同时输出OpenCV与LaMa修复结果")
    with gr.Row():
        img_in = gr.Image(type="numpy", label="上传壁画图片")
        thresh_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="AI识别阈值 threshold")
    btn = gr.Button("开始修复")
    out_img = gr.Image(label="对比结果")
    btn.click(pipeline, inputs=[img_in, thresh_slider], outputs=[out_img])

if __name__ == "__main__":
    demo.launch(share=True)

