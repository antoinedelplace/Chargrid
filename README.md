# Chargrid model : Extraction of meaningful instances from document images
_Author: Antoine DELPLACE_  
_Last update: 26/03/2020_

This repository corresponds to my implementation of "__Chargrid: Towards Understanding 2D Documents__" by A. R. Katti et al. The data used to test the model comes from the ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction.

## Data description
The ICDAR 2019 dataset is composed of 690 images of receipts along with, for each invoice, a json file containing 4 classes (the company name, the date, the store address and the total amount) and the true bounding boxes of strings in the bill. 

Because the inputs have purposely "poor paper quality, poor ink and printing quality; low resolution scanner and scanning distortion", only 682 images were kept after the Optical Character Recognition (OCR). 

The images are of different shapes and resolutions. 

The number of classes successfully extracted from the bounding boxes with the json file are presented in the following table :
| Number of detected classes | 0 | 1 | 2 | 3  | 4   |
|----------------------------|---|---|---|----|-----|
| Number of input images     | 8 | 7 | 3 | 34 | 630 |

## Method description
- The first step is to preprocess the dataset in order to have homogeneous and trainable inputs (chargrid) and ground truths.
- Once chargrids are ready, they are used to train the model composed of :
    - a VGG encoder
    - a Semantic Segmentation Decoder with skip connections
    - a Bounding Box Regression Decoder with skip connections

## Usage

### Dependencies
- Python 3.6.8
- Numpy 1.16.2
- Pandas 0.24.2
- Matplotlib 3.1.1
- Scikit-learn 0.20.3 -- `preprocessing.py`
- Pytesseract 0.3.3 -- `preprocessing.py`
- Tensorflow 2.0.0 -- `network.py`
- Scikit-image 0.15.0 -- `preprocessing_ter.py` and `network.py`

### File description
1. `preprocessing.py` is the first preprocessing program designed to:
    - generate Chargrids from input images thanks to Tesseract
    - extract bounding boxes for each class from the ground truth files
    - generate class segmentation from the class bounding boxes
    - reduce the size of images by removing empty rows and empty columns

2. `preprocessing_bis.py` is the second preprocessing program that reduces the size of each image as much as possible without losing too much information.

3. `preprocessing_ter.py` is the third and final preprocessing program designed to:
    - resize all images to the same shape
    - convert all inputs and ground truth files to one-hot encoding

4. `network.py` is the main program to train and test the chargrid model.

## Results
[TODO]

## Possible improvements
- Remove one-hot encoding and use categorical cross-entropy (more memory-effective)
- Implement focal cross-entropy loss

## References
1. A. R. Katti et al. "Chargrid: Towards Understanding 2D Documents", _EMNLP 2018_, September 2018. [arXiv:1809.08799](https://arxiv.org/abs/1809.08799)
2. Z. Huang et al. ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction, _ICDAR, 2019_, 2019. [link](https://rrc.cvc.uab.es/?ch=13)