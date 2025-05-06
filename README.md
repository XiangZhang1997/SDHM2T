# SDHM<sup>2</sup>T——A Scale Direction Heteroid Micro to Macro Transition Network for Retinal Vessel Segmentation
![](https://img.shields.io/badge/license-MIT-blue)
The official implementation of SDHM<sup>2</sup>T

## Overview
SDHM<sup>2</sup>T is a project for vessel segmentation using deep learning techniques. This repository contains scripts for training, testing, and evaluating models on the retinal datasets.

## Scripts

### `train.py`
This script is used for training the model on the DRIVE dataset.

**Usage:**
1. Prepare your dataset in the appropriate format.
2. Configure the training parameters in the script or via command-line arguments.
3. Run the script using the following command:
    ```bash
    python /home/s1/ZX/job/Vessel/train.py -dp "/home/s1/ZX/job/Vessel/datasets/DRIVE" --val
    ```

### `test.py`
This script is used for evaluating the trained model on test images.

**Usage:**
1. Ensure that you have a trained model available.
2. Configure the testing parameters in the script or via command-line arguments.
3. Run the script using the following command:
    ```bash
    python /home/s1/ZX/job/Vessel/test.py -dp "/home/s1/ZX/job/Vessel/datasets/DRIVE" -wp "/home/s1/ZX/job/Vessel/pretrained_weights/DRIVE/SDHM2T/checkpoint-epoch20.pth" --show
    ```

### `metrics.py`
This script calculates various evaluation metrics for the segmentation results.

**Usage:**
1. Ensure that the segmentation results and ground truth maps are available.
2. Run the script to calculate metrics using the following command:
    ```bash
    python /home/s1/ZX/job/Vessel/c_metrics.py -dp "/home/s1/ZX/job/Vessel/datasets/DRIVE"
    ```

## Example
1. **Training**: Prepare your dataset, then run training:
    ```bash
    python /home/s1/ZX/job/Vessel/train.py -dp "/home/s1/ZX/job/Vessel/datasets/DRIVE" --val
    ```
2. **Testing**: Evaluate your trained model:
    ```bash
    python /home/s1/ZX/job/Vessel/test.py -dp "/home/s1/ZX/job/Vessel/datasets/DRIVE" -wp "/home/s1/ZX/job/Vessel/pretrained_weights/DRIVE/SDHM2T/checkpoint-epoch20.pth" --show
    ```
3. **Metrics**: Calculate performance metrics:
    ```bash
    python /home/s1/ZX/job/Vessel/c_metrics.py -dp "/home/s1/ZX/job/Vessel/datasets/DRIVE"
    ```

## Requirements
- Python 3.7.4
- PyTorch 1.8.0
- torchvision 0.9.0
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
