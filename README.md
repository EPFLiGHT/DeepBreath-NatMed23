<p align="center" style="display:flex; justify-content:space-between; width:100%;">
    <a href="https://www.epfl.ch"><img src="assets/epfl_logo.png" alt="EPFL" style="height:90px; object-fit:cover;"></a>
    <a href="https://www.hug.ch/"><img src="assets/hug_logo.png" alt="HUG" style="height:90px; object-fit:cover;"></a>
    <a href="https://www.epfl.ch/labs/mlo/igh-intelligent-global-health/"><img src="assets/igh_logo.png" alt="iGH" style="height:90px; object-fit:cover;"></a>
</p>


DeepBreath-NatMed23
==============================

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Lung auscultation is an essential clinical exam in the evaluation of respiratory diseases, but it is subjective and relies on non-specific nomenclature. Computer-aided analysis has the potential to standardize and automate evaluation, making it more accessible to low-resource and remote care settings. In this repository, we present DeepBreath, a deep learning model that recognizes audible signatures of pediatric respiratory diseases.

## Introduction

The DeepBreath model was trained on 4552 digital lung auscultation recordings from 572 pediatric outpatients (35.9 hours of audio) from six outpatient departments across five countries (Switzerland, Brazil, Senegal, Cameroon, and two sites in Morocco). Cases with acute respiratory illnesses (71%, n = 407/572) comprised radiographically confirmed pneumonia or clinically diagnosed wheezing disorders (bronchitis/asthma), and bronchiolitis. The remaining 29% were healthy controls.

DeepBreath comprises a convolutional neural network followed by a logistic regression classifier, aggregating estimates on recordings from eight thoracic sites into a single prediction at the patient-level. Temporal attention identifies the most informative regions of the recording, allowing an interpretable exploration of predictions.

## Results

To ensure objective performance estimates on model generalisability, DeepBreath is trained on patients from two sites (Switzerland and Brazil), and results then reported on an internal 5-fold cross validation as well as externally validated (extval) on recordings from four remaining sites (Senegal, Cameroon, and two in Morocco). The model was able to discriminate healthy and pathological breathing with an Area Under the Receiver-Operator Characteristic (AUROC) of 0.93 (standard deviation [SD] ±0.01 on internal validation). Similarly promising results were obtained for pneumonia (AUROC 0.75 ±0.10), wheezing disorders (AUROC 0.91 ±0.03), and bronchiolitis (AUROC 0.94 ±0.02). Extval AUROCs were 0.89, 0.74, 0.74 and 0.87 respectively. All either matched or were significant improvements on a clinical baseline model using age and respiratory rate.

Attention plots show a clear alignment between model prediction and independently annotated respiratory cycles, providing evidence that DeepBreath extracts physiologically meaningful representations from auscultations.

## Usage

This repository contains the code for training and testing the DeepBreath model on lung auscultation recordings. It includes the data preprocessing and augmentation scripts, the model architecture, training and testing scripts, and the attention plot generation script. The code was written in Python (version 3.7.4) and relies on several third-party packages, which are listed in the `requirements.txt` file.

## Installation

   1. Clone the repository
   2. Install the required packages using pip: `pip install -r requirements.txt`

## Data

The lung auscultation recordings used to train and test the DeepBreath model are not included in this repository due to privacy concerns. However, the data preprocessing and augmentation scripts are included, so it is possible to preprocess your own lung auscultation recordings and train the model on them.

## Training

To train the DeepBreath model on your own lung auscultation recordings, move to the `src/` folder and follow these steps:

   3. Preprocess and augment the data using the `prepare_data.py` script
   4. Train the model using the `train_audio_classifier.py` script

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── out                
    │   ├── aggregate      <- 
    │   ├── features       <- 
    │   └── models         <- Trained and serialized models
    │
    ├── report             <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── data
        │   ├── dataset.py
        │   └── sampler.py
        │
        ├── models
        │   ├── Cnn10Att.py
        │   ├── Cnn10.py
        │   ├── Cnn6.py
        │   └── SampleModel.py
        │
        ├── prepare_data.py
        │
        ├── preprocessing
        │   ├── features.py
        │   └── helpers.py
        │
        ├── train_audio_classifier.py
        │
        ├── training
        │   ├── pipeline.py
        │   └── sample_fit.py
        │
        └── utils
           ├── config.py
           └── constants.py
--------

## License

The code in this repository is released under the Apache-2.0 license. See the `LICENSE` file for more details.

## Contact

If you have any questions or comments about this code, please contact **TODO**.

## Citation

If you use this code in your research, please cite the following paper: **TODO**.
