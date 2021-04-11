# VSpSR: Explorable Super-Resolution via Variational Sparse Representation


This repository is an official PyTorch implementation of the VSpSR from **NTIRE21, Learning SR Space**.

We provide scripts for reproducing all the results. You can train your model from scratch, or use a pre-trained model to get different sr images.

## Requirements
VSpSR is built in Python 3.6 using PyTorch 1.7.1. Use the following command to install the requirements:
```
pip install -r requirements.txt
```

## Quickstart (Demo)
You can test our super-resolution algorithm with your images. Place your images in any folder you like. (like ``test``).

Run the script in ``src`` folder. Before you run the demo, please uncomment the appropriate line in ```demo.sh``` that you want to execute.
```bash
cd VSpSR      # You are now in */VSpSR
sh demo.sh
```

You can find the result images from ```output``` folder.

## How to train VSpSR
We used DIV2K from NTIRE21 dataset to train our model. Please download it from [tr_1X](https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-tr_1X.zip), 
[tr_4X](https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-tr_4X.zip), [tr_8X](https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-tr_8X.zip), 
[va_1X](https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-va_1X.zip), 
[va_4X](https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-va_4X.zip), [va_8X](https://data.vision.ee.ethz.ch/alugmayr/NTIRE2021/DIV2K-va_8X.zip).

Download all the files to any place you want(like ``NTIRE2021``). Then, change the ```dataset_root``` argument in ```demo.sh``` to the place where DIV2K images are located.

You can train VSpSR by yourself. All scripts are provided in the ``demo.sh``.

One 12-GB Titan X GPU is used for training VSpSR. Training takes about 13 hours.

```bash
cd VSpSR       # You are now in */VSpSR
sh demo.sh
```

