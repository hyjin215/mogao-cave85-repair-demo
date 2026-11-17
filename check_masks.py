
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "muralseg"

for split in ["train", "val", "test"]:
    img_dir = DATA / split / "images"
    mask_dir = DATA / split / "labels"

    images = sorted(img_dir.glob("*.png"))

    print(f"\n{split}: 找到 {len(images)} 张图片")

    if split != "train":
        continue

    for i, img_path in enumerate(images[:3]):
        mask_path = mask_dir / img_path.name

        if not mask_path.exists():
            print("缺少Mask:", img_path.name)
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), 0)

        if img is None or mask is None:
            print("读取失败:", img_path.name)
            continue

        print("\n文件:", img_path.name)
        print("图片尺寸:", img.shape)
        print("Mask尺寸:", mask.shape)
        print("白色区域比例:",
              round(np.mean(mask > 127) * 100, 2), "%")

        mask_color = cv2.cvtColor(
            mask, cv2.COLOR_GRAY2BGR
        )

        overlay = img.copy()
        overlay[mask > 127] = (
            0.5 * overlay[mask > 127]
            + 0.5 * np.array([0, 0, 255])
        ).astype(np.uint8)

        preview = np.hstack([
            cv2.resize(img, (400, 400)),
            cv2.resize(mask_color, (400, 400)),
            cv2.resize(overlay, (400, 400))
        ])

        cv2.imwrite(
            str(BASE / f"mask_check_{i+1}.jpg"),
            preview
        )

print("\n检查完成！")
