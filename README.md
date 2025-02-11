# vil_images_generation
## Project description

The goal of project is to generate the missing “vil” images based on a sequence of “vis”, “ir069”, and “ir107” images from a single storm.

<img src="images/task.jpg" alt="goal" width="250" height="100">


## Dataset
This project have access to a generative model that has been trained to produce realistic-looking MRI images of patient's heads. So I used the provided image-generation network to create a dataset of brain images as the training dataset which is called 'train_data_3000.pt'. This training dataset includes 3000 samples.

## Network construction
Unet combined with multi-head self-attention mechanism

## Model performance
| **Dataset** | **LOSS** | **SSIM** |
| :------------------ | :---: | :---: |
| *training* | 0.0003 | 0.9568 |
| *validation* | 0.0004 | 0.9507 |