# Chargrid
_Author: Antoine DELPLACE_  
_Last update: 23/03/2020_

This repository corresponds to my implementation of "__Chargrid: Towards Understanding 2D Documents__" by A. R. Katti et al. The data used to test the model comes from the ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction.

## Data description
[TODO]

## Method description
[TODO]

## Usage

### Dependencies
- Python 3.6.8
- Numpy 1.16.2
- Pandas 0.24.2
- Matplotlib 3.1.1
- Pytesseract 0.3.3 -- `preprocessing.py`
- Tensorflow 2.0.0 -- `network.py`
[TODO]

### File description
1. The file `preprocessing.py` performs an optical character recognition on the image dataset. It creates a chargrid for each input image.

2. The file `preprocessing_bis.py` reduces each chargrid to improve the training speed.

3. The file `preprocessing_ter.py` creates the scaled 1-hot encoding input images to train.

4. The file `network.py` is in development.

## Results
[TODO]

## References
1. A. R. Katti et al. "Chargrid: Towards Understanding 2D Documents", _EMNLP 2018_, September 2018. [arXiv:1809.08799](https://arxiv.org/abs/1809.08799)