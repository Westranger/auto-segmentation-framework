# Auto-Segmentation Framework

This repository implements an **end-to-end workflow for semantic segmentation** that automatically finds a minimal U-Net
architecture with high performance. It uses **Optuna** to explore different depths, filter sizes and channel counts while
penalizing model complexity and latency. The aim is to produce a compact network suitable for deployment on **edge devices**.

The Framework slices the input images into small tiles and uses only a small subset of the tiles to train the model and
to find the smallest possible architecture while trying to keep the accuracy high. When a net was found by optuna it is
applied to the full size images of the test set. But it is also applied tiles/patchwise in an moving window manner. 
Afterward the accuracy in calculated. 

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
```
afterward if python is installed one can oly install all required packages with
```
pip install -r requirements.txt
```
or directly set up the virtual environment with 
```
sh setup_env.sh
```

which also installs the required python version

Python 3.9+ and PyTorch 2.0+ are recommended.

## Usage

1. change images and masks paths in `main_search.py`
2. adjust parameters like num representatives, tile size, and overlap
3. run `main_search.py` which will go through the whole process of finding the optimal UNet
5. Export the trained model to ONNX is done automatically

## Example Performance

On the [Kaggle Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset) dataset a network withe parameters :
- patchsize = 128
- number layer = 2
- encoder_specs = ((3, 16), (3, 64))
- representatives = 200

achieved a **cross-validation score of 0.9995** on the training set and on the testset (full size images) a **score of 0.9990**.
The original 700 full size images were tiled with a moving window approach to 140k tiles (in memory) afterward 200 
representatives were selected on which the net was trained. In the end the full size images were taken and the small 
result net wa applied with a moving window approach on the full size images.
In simple word net the was trained on small patches of the original full images and later on patchwise applied the full
images of the testset again


## Future improvements

- [ ] externalize configuration parameters and hard coded path to a `config.yml`
- [ ] add config parameter which allows to specify the GPU RAM size and which internally selects the batch size
      for training automatically
- [ ] change `num_prepresentatives` to a new variable `num_prepresentatives_per_cluster` which internally selects
      only `n` representatives per cluster after computing the required amount of clusters  

## License

This project is for non-commercial use. Commercial usage requires prior permission.
