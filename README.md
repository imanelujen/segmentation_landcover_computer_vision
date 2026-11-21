
# LandCover.ai - Semantic Segmentation

## Project Overview

This project performs semantic segmentation of aerial images
using a U-Net deep learning architecture.

The model was trained using the LandCover.ai dataset.

## Classes

| ID | Class | Color |
|---|---|---|
| 0 | Background | Black |
| 1 | Building | Red |
| 2 | Woodland | Green |
| 3 | Water | Blue |
| 4 | Road | Yellow |

## Model

- Architecture: U-Net
- Input channels: 3 (RGB)
- Input size: 512 x 512
- Number of classes: 5
- Best checkpoint: `checkpoints/unet_best.pth`
- Best epoch: 19
- Validation loss: 0.3301

## Test Results

| Metric | Score |
|---|---:|
| Pixel Accuracy | 88.53% |
| Mean IoU | 50.10% |
| Mean Dice | 61.42% |

## Project Structure

```text
segmentation_landcover/
├── app/
├── checkpoints/
│   └── unet_best.pth
├── src/
│   ├── models/
│   │   └── unet.py
│   └── inference/
│       ├── __init__.py
│       └── predict.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Backend Usage

```python
from src.inference.predict import load_model, predict_image
from PIL import Image

model, device = load_model()

image = Image.open("example.png")

result = predict_image(image, model, device)

print(result["class_percentages"])
```

## Output

The prediction function returns:

- Original image
- Predicted segmentation mask
- Colored segmentation mask
- Overlay
- Percentage of each predicted class

## Dataset Limitation

LandCover.ai provides four annotated object classes:
Buildings, Woodlands, Water and Roads, plus Background.

Agriculture and Bare Soil are not separate annotated classes
in this dataset.
