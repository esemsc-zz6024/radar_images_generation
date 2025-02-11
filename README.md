# radar_images_generation
## Project description


The goal of this project is to generate the missing “vil” images based on a sequence of “vis”, “ir069”, and “ir107” images from a single storm.
![task](images/task.jpg)

## Dataset
There are 4 types of satellite images: visible, water vapour (infrared), cloud/surface temperature (infrared), vertically integrated liquid (radar). The whole dataset includes 800 example storms over the entire US.


## Network construction
Unet combined with multi-head self-attention mechanism

## Model performance
| **Dataset** | **LOSS** | **SSIM** |
| :------------------ | :---: | :---: |
| *training* | 0.0003 | 0.9568 |
| *validation* | 0.0004 | 0.9507 |