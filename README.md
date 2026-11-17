# Mogao Cave 85 — AI-Assisted Virtual Restoration

**An AI-Assisted Workflow for Mural Damage Segmentation, Human-in-the-Loop Review, and Digital Restoration**

---

## 1. Introduction

This project presents an AI-assisted digital restoration workflow inspired by the conservation and restoration of Mogao Cave 85.

The system integrates deep learning-based damage segmentation, manual mask correction, human review, and image inpainting to demonstrate how artificial intelligence can assist the preliminary stages of mural restoration.

The workflow combines **U-Net++**, **OpenCV**, and **LaMa** to identify potential damaged regions and generate visual restoration results.

Rather than allowing an AI model to make restoration decisions independently, this project adopts a **Human-in-the-Loop approach**. Automatically generated damage masks can be manually corrected and reviewed before being used for restoration.

The project is designed as an educational and technical demonstration. It does not replace professional archaeological research, conservation expertise, or expert restoration decisions.

## 2. Key Features

- **Automatic Damage Segmentation:** Uses a trained U-Net++ model to identify potential damaged regions in mural images.
- **Visual Mask Editing:** Provides an interactive OpenCV interface for manually adding or removing damaged regions.
- **Human-in-the-Loop Review:** Allows users to inspect and confirm the damage mask before restoration.
- **Multiple Restoration Methods:** Combines OpenCV Telea Inpainting and LaMa for comparative restoration.
- **Restoration Comparison:** Generates visual comparisons between the original image, damage mask, and restoration results.
- **Review Records:** Saves confirmed masks and restoration metadata for subsequent inspection.
- **Manual Fallback:** Supports manual damage annotation when the trained segmentation checkpoint is unavailable.

## 3. System Workflow

The system follows a six-stage workflow:

```text
       Input Mural Image
               |
               v
    U-Net++ Damage Segmentation
               |
               v
      Automatic Damage Mask
               |
               v
       Manual Mask Correction
               |
               v
       Human Review & Confirmation
               |
               v
       Confirmed Damage Mask
               |
        +------+------+
        |             |
        v             v
   OpenCV Telea      LaMa
    Inpainting    Inpainting
        |             |
        v             v
   OpenCV Result   LaMa Result
        |             |
        +------+------+
               |
               v
      Restoration Comparison
               |
               v
        Review Record
```

### Stage 1 — Input Image

Users place mural images into the `inputs/` directory.

Supported formats:

- JPG
- JPEG
- PNG
- BMP

The program supports processing individual images or selecting from multiple available images.

### Stage 2 — Automatic Damage Segmentation

A trained U-Net++ model is used to predict potential damaged regions.

The segmentation component produces:

- `mask_auto.png` — Automatically predicted binary damage mask.
- `mask_probability.png` — Pixel-level damage probability map.

The default prediction threshold is `0.50`.

If the trained checkpoint is unavailable or automatic prediction fails, the workflow can fall back to manual mask annotation.

### Stage 3 — Manual Mask Correction

Users can inspect and modify the predicted mask through an interactive OpenCV interface.

| Control | Function |
|---|---|
| Left mouse button | Mark damaged regions |
| Right mouse button | Erase marked regions |
| `+` / `-` | Adjust brush size |
| `R` | Clear the mask |
| `S` | Finish editing |
| `ESC` | Exit the editing interface |

This stage allows users to correct false positives, false negatives, and inaccurate damage boundaries.

### Stage 4 — Human Review and Confirmation

After editing, the preliminary mask is saved as:

`mask_pending_review.png`

The program then requests explicit confirmation in the terminal.

Enter:

```text
YES
```

to approve the mask and proceed with restoration.

The confirmed mask is saved separately to distinguish it from the original automatic prediction.

Key files:

- `mask_expert_confirmed.png`
- `mask.png`
- `original.png`

The review stage helps ensure that unverified AI predictions are not directly treated as final restoration masks.

### Stage 5 — Image Inpainting

The confirmed damage mask is used by two restoration methods.

**A. OpenCV Telea Inpainting**

OpenCV Telea Inpainting provides a traditional image inpainting baseline.

Output:

`result_opencv.png`

**B. LaMa Inpainting**

The project integrates LaMa (Large Mask Inpainting) through the `simple-lama-inpainting` package.

LaMa generates reconstructed image content within the masked regions. The pipeline preserves original pixels outside the confirmed mask.

Output:

`result_lama.png`

If LaMa fails to initialize or generate a result, the OpenCV restoration result remains available.

### Stage 6 — Restoration Comparison

The system generates a comparison image:

`Final_Comparison.jpg`

This allows users to examine the original mural, confirmed damage mask, and restoration outputs.

A JSON review record is also generated:

`review_record.json`

## 4. Technologies

| Component | Technology |
|---|---|
| Programming Language | Python |
| Damage Segmentation | U-Net++ |
| Deep Learning Framework | PyTorch |
| Segmentation Library | Segmentation Models PyTorch |
| Image Processing | OpenCV |
| Traditional Inpainting | OpenCV Telea |
| Deep Learning Inpainting | LaMa |
| LaMa Integration | simple-lama-inpainting |
| Image Handling | Pillow |

## 5. Repository Structure

The repository contains the project source code and configuration files. The original third-party dataset is not bundled with the repository.

```text
Mogao85_AutoRestoration/
│
├── inputs/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── checkpoints/
│   └── README.md
│
├── muralseg/
│   └── README.md
│
├── run_pipeline.py
├── damage_segmentation.py
├── train_unetplusplus.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

**Directory descriptions:**

- `inputs/` — Stores mural images to be processed.
- `outputs/` — Stores generated masks, restoration images, comparison images, and review records.
- `checkpoints/` — Stores trained U-Net++ model weights when available.
- `muralseg/` — Contains dataset preparation instructions and the expected dataset structure.
- `run_pipeline.py` — Main restoration pipeline.
- `damage_segmentation.py` — Damage segmentation and prediction module.
- `train_unetplusplus.py` — U-Net++ model training script.

The actual repository structure may include additional supporting scripts and configuration files.

## 6. Installation

### Requirements

- Python 3.10 or later
- PyTorch
- OpenCV
- Pillow
- Segmentation Models PyTorch
- simple-lama-inpainting

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Mogao85_AutoRestoration.git

cd Mogao85_AutoRestoration
```

Replace `YOUR_USERNAME` with the actual GitHub username.

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

PyTorch and Torchvision should be installed with versions compatible with the operating system and CPU/CUDA environment.

Refer to the official PyTorch installation guide if a specific hardware configuration is required.

## 7. Dataset Preparation

### MuralDH Segmentation Subset

The U-Net++ damage segmentation component uses the MuralDH Segmentation Subset, a reorganized dataset derived from the original MuralDH dataset.

The subset provides 961 images with pixel-level damage annotations.

| Dataset Split | Number of Images |
|---|---:|
| Training | 761 |
| Validation | 100 |
| Testing | 100 |
| **Total** | **961** |

The images are provided in PNG format at 512 × 512 resolution.

The binary annotation masks follow this convention:

- `255` — Damaged regions.
- `0` — Intact regions.

### Dataset Download Sources

**1. MuralDH Segmentation Subset (Reorganized)**

Repository:

https://github.com/bearybeary-beary/MuralDH-Segmentation

The dataset download instructions and Google Drive link are provided in the original dataset repository.

**2. Original MuralDH Dataset**

Official repository:

https://github.com/tearsheaven/MuralDH

Original publication:

Xu et al., *A comprehensive dataset for digital restoration of Dunhuang murals*, Scientific Data, 2024.

### Important Dataset Notice

The original mural images and annotation masks are not included in this GitHub repository.

Users should obtain the dataset directly from its original source and comply with the applicable copyright, licensing, and usage conditions.

The dataset is intended for academic and educational research. Commercial use is restricted under the conditions stated by the original MuralDH publication.

This project does not claim ownership of the original dataset or its annotations.

### Expected Dataset Structure

After downloading and extracting the dataset, organize the files as follows:

```text
muralseg/
│
├── train/
│   ├── images/
│   └── labels/
│
├── val/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

Please ensure that each image has a corresponding annotation mask with a matching filename.

## 8. Model Checkpoint

The automatic damage segmentation component requires a trained U-Net++ checkpoint.

The expected location is:

```text
checkpoints/
└── unetplusplus_best.pth
```

If a compatible checkpoint is available, place it in the directory above.

If the checkpoint is not included in the repository, users can train their own model using the provided training script, provided that the required dataset and dependencies have been prepared.

Without a checkpoint, the system can still operate through its manual mask annotation workflow.

**Note:** Model weights are treated separately from project source code. Their redistribution should only take place after checking the applicable dataset and model licensing conditions.

## 9. Running the Restoration Pipeline

### Step 1 — Prepare Input Images

Place one or more mural images in:

```text
inputs/
```

Example:

```text
inputs/
├── mural_01.jpg
├── mural_02.png
└── mural_03.jpg
```

### Step 2 — Run the Program

```bash
python run_pipeline.py
```

If multiple images are available, select the image to process when prompted.

### Step 3 — Inspect and Correct the Mask

Review the automatically generated damage mask.

Use the OpenCV interface to add or erase damaged regions as necessary.

Press `S` to finish editing.

### Step 4 — Confirm the Mask

Enter `YES` in the terminal when prompted.

The restoration process will proceed after confirmation.

### Step 5 — View Restoration Results

Generated files are saved in:

```text
outputs/
```

**Important:** The current pipeline uses fixed output filenames. Processing another image may overwrite previous results. Move or rename existing output files if you want to preserve earlier results.

## 10. Output Files

| Output File | Description |
|---|---|
| `mask_auto.png` | Automatically predicted damage mask |
| `mask_probability.png` | Damage probability map |
| `mask_pending_review.png` | Preliminary mask awaiting confirmation |
| `mask_expert_confirmed.png` | Human-confirmed damage mask |
| `mask.png` | Confirmed mask used for restoration |
| `original.png` | Original input image |
| `result_opencv.png` | OpenCV Telea restoration result |
| `result_lama.png` | LaMa restoration result, if successfully generated |
| `Final_Comparison.jpg` | Visual comparison of restoration results |
| `review_record.json` | Review and processing metadata |

## 11. Model Training

The project includes a U-Net++ training script for damage segmentation.

Before training, prepare the MuralDH segmentation dataset according to Section 7.

Run:

```bash
python train_unetplusplus.py
```

The training script uses the dataset organized under `muralseg/`.

The trained checkpoint can subsequently be placed at:

```text
checkpoints/unetplusplus_best.pth
```

for use by the main restoration pipeline.

Training performance and segmentation quality depend on dataset quality, training configuration, and hardware resources.

## 12. Related Research and Third-Party Resources

### MuralDH

The MuralDH dataset provides research resources for digital restoration of Dunhuang murals.

Official repository:

https://github.com/tearsheaven/MuralDH

### UASTINet

UASTINet is related research on uncertainty-aware joint structure-texture inpainting for Dunhuang murals.

Repository:

https://github.com/Muralhub/UASTINet

UASTINet is acknowledged as related research. It is not integrated into the current restoration pipeline.

### LaMa

LaMa (Large Mask Inpainting) is used as the deep-learning-based image restoration method.

Original repository:

https://github.com/advimman/lama

Python integration:

https://github.com/enesmsahin/simple-lama-inpainting

### Segmentation Models PyTorch

Repository:

https://github.com/qubvel-org/segmentation_models.pytorch

Third-party software, datasets, pretrained weights, and research implementations remain subject to their respective licenses and usage conditions.

## 13. Human-in-the-Loop Design

A central principle of this project is that AI-generated predictions should support, rather than replace, human judgment.

The U-Net++ model identifies potential damaged regions. However, automatic predictions may contain false positives, false negatives, or inaccurate boundaries.

The system therefore incorporates:

1. **Automatic Identification:** AI-assisted detection of potential damage.
2. **Manual Correction:** Users can modify predicted damage masks.
3. **Human Confirmation:** The mask must be explicitly approved before restoration.
4. **Result Inspection:** Different inpainting results can be compared visually.

This approach provides a more transparent and reviewable restoration workflow.

The generated images are experimental digital visualizations, not verified reconstructions of the historically accurate original mural.

## 14. Limitations

- Segmentation performance depends on the quality and coverage of the training dataset.
- The trained model may not generalize to all mural styles, textures, lighting conditions, or deterioration patterns.
- Automatic predictions require human inspection and correction.
- Inpainting algorithms generate plausible image content but cannot guarantee historically accurate reconstruction.
- LaMa may require additional model downloads and compatible runtime dependencies.
- Restoration quality depends heavily on the accuracy of the confirmed damage mask.
- The project is an educational prototype and has not been validated as a professional conservation system.

## 15. Future Development

Potential future improvements include:

- Improving damage segmentation accuracy through additional training data.
- Supporting more types of mural deterioration.
- Enhancing the interactive mask annotation interface.
- Developing a graphical user interface for non-technical users.
- Integrating additional restoration models for comparative evaluation.
- Introducing more comprehensive restoration quality assessment.
- Improving review records and expert annotation management.

## 16. Acknowledgements

This project acknowledges the following research and software resources:

- **MuralDH** — A dataset for digital restoration research on Dunhuang murals.
- **MuralDH Segmentation Subset** — Reorganized pixel-level damage annotations for mural segmentation research.
- **UASTINet** — Related research on uncertainty-aware structure-texture inpainting for Dunhuang murals.
- **LaMa** — Large Mask Inpainting.
- **Simple-Lama-Inpainting** — Python integration for LaMa.
- **Segmentation Models PyTorch** — Deep-learning segmentation framework.
- **OpenCV** — Image processing and Telea inpainting implementation.

We sincerely acknowledge the original researchers and developers for making their research and implementations available.

### Dataset Citation

If you use the original MuralDH dataset in your research, please cite the original publication:

> Xu, Z., Yang, Y., Fang, Q., et al. (2024). A comprehensive dataset for digital restoration of Dunhuang murals. *Scientific Data*, 11(1), 955.

Original repository:

https://github.com/tearsheaven/MuralDH

**BibTeX:**

```bibtex
@article{xu2024comprehensive,
  title={A comprehensive dataset for digital restoration of Dunhuang murals},
  author={Xu, Zishan and Yang, Yuqing and Fang, Qianzhen and others},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={955},
  year={2024}
}
```

## 17. License

The licensing terms of this project and its third-party components should be considered separately.

The original project source code and third-party resources are not automatically covered by the same license.

Unless an explicit license is included, no additional permission to redistribute the original dataset, annotations, or third-party model weights is granted.

Please refer to the respective original repositories and licensing terms before using or redistributing third-party materials.

## 18. Disclaimer

This project is developed for educational, academic, and research demonstration purposes.

AI-generated segmentation and inpainting results are experimental visual outputs. They should not be considered authoritative historical reconstructions or substitutes for professional archaeological research, mural conservation, or expert restoration decisions.