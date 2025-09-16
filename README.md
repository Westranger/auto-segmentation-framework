# Auto-Segmentation Framework

This repository implements an **end-to-end workflow for semantic segmentation** that automatically finds a minimal U-Net architecture with high performance. It uses **Optuna** to explore different depths, filter sizes and channel counts while penalizing model complexity and latency. The aim is to produce a compact network suitable for deployment on **edge devices**.

## Features

- **Representative tile selection** via embedding-based clustering to reduce datasets.
- **Optuna-based architecture search** balancing accuracy and model complexity by searching through parameters like number of layers, channels and kernel sizes.
- **Cross-validation training** with robust evaluation.
- **ONNX export** for deployment.
- **Configurable pipeline parameters**.

## Installation

```bash
git clone https://github.com/Westranger/auto-segmentation-framework.git
cd auto-segmentation-framework
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.0+ are recommended.

## Usage

1. change images and masks paths in `main_searcher.py`
2. adjust parameters like num representatives, tile size, and overlap
3. run `main_searcher.py` which will go through the whole process of finding the optimal UNet
5. Export the trained model to ONNX is done automatically

## Example Performance

On the [Kaggle Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset) dataset a network withe parameters :
- patchsize = 128
- number layer = 2
- encoder_specs = ((3, 16), (3, 64))
- representatives = 200

achieved a **cross-validation score of 0.9995**

## License

This project is for non-commercial use. Commercial usage requires prior permission.
