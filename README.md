<p align="center" style="display:flex; justify-content:space-between; width:100%;">
    <a href="https://www.epfl.ch"><img src="assets/epfl_logo.png" alt="EPFL" style="height:80px; object-fit:cover;"></a>
    <a href="https://www.hug.ch/"><img src="assets/hug_logo.png" alt="HUG" style="height:80px; object-fit:cover;"></a>
    <a href="https://www.epfl.ch/labs/mlo/igh-intelligent-global-health/"><img src="assets/igh_logo.png" alt="iGH" style="height:80px; object-fit:cover;"></a>
</p>


DeepBreath-NatMed23
==============================

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/epfl-iglobalhealth/DeepBreath-NatMed23/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is code accompanying the publication

> J. Heitmann et al. [DeepBreath---automated detection of respiratory pathology from lung auscultation in 572 pediatric outpatients across 5 countries](https://www.nature.com/articles/s41746-023-00838-3).
*npj Digital Medicine* **6** (2023)

Lung auscultation is an essential clinical exam in the evaluation of respiratory diseases, but it is subjective and relies on non-specific nomenclature. Computer-aided analysis has the potential to standardize and automate evaluation, making it more accessible to low-resource and remote care settings. In this repository, we present DeepBreath, a deep learning model that recognizes audible signatures of pediatric respiratory diseases.

## Introduction

The DeepBreath model was trained on 4552 digital lung auscultation recordings from 572 pediatric outpatients (35.9 hours of audio) from six outpatient departments across five countries (Switzerland, Brazil, Senegal, Cameroon, and two sites in Morocco). Cases with acute respiratory illnesses (71%, n = 407/572) comprised radiographically confirmed pneumonia or clinically diagnosed wheezing disorders (bronchitis/asthma), and bronchiolitis. The remaining 29% were healthy controls.

DeepBreath comprises a convolutional neural network followed by a logistic regression classifier, aggregating estimates on recordings from eight thoracic sites into a single prediction at the patient-level. Temporal attention identifies the most informative regions of the recording, allowing an interpretable exploration of predictions.

## Results

To ensure objective performance estimates on model generalisability, DeepBreath is trained on patients from two sites (Switzerland and Brazil), and results then reported on an internal 5-fold cross validation as well as externally validated (extval) on recordings from four remaining sites (Senegal, Cameroon, and two in Morocco). The model was able to discriminate healthy and pathological breathing with an Area Under the Receiver-Operator Characteristic (AUROC) of 0.93 (standard deviation [SD] ±0.01 on internal validation). Similarly promising results were obtained for pneumonia (AUROC 0.75 ±0.10), wheezing disorders (AUROC 0.91 ±0.03), and bronchiolitis (AUROC 0.94 ±0.02). Extval AUROCs were 0.89, 0.74, 0.74 and 0.87 respectively. All either matched or were significant improvements on a clinical baseline model using age and respiratory rate.

Attention plots show a clear alignment between model prediction and independently annotated respiratory cycles, providing evidence that DeepBreath extracts physiologically meaningful representations from auscultations.

## Usage

This repository contains the code for training and testing the DeepBreath model on lung auscultation recordings. It includes the data preprocessing and augmentation scripts, the model architecture, training and testing scripts, and the code to generate the attention plots. The code was written in **Python 3.10** and relies on several third-party packages, which are listed in the `requirements.txt` file.

## Installation

   - Clone the repository
   - Install the required packages using pip: `pip install -r requirements.txt`

## Data

Anonymized data are available upon reasonable request which matches the intention to improve the diagnosis of paediatric respiratory disease in resource-limited settings. The audio used in the study are not publicly available to protect participant privacy. Unlimited further use is not permissible from the informed consent. The data preprocessing and augmentation scripts are included, so it is possible to preprocess your own lung auscultation recordings and train the model on them.

## Training

Move to the `src/` folder for data preparation and model training. 

The data that were used to train the DeepBreath models were preprocessed using the `prepare_data.py` script. The script was designed specifically for the DeepBreath dataset, it contains dataset-specific arguments (e.g. `--asthmoscope_locations` and `--pneumoscope_locations`) and should not be used to preprocess another dataset unless modified accordingly. 

To train a DeepBreath model using the `train_audio_classifier.py` script, you need to provide a configuration file in JSON format. The configuration file specifies the data paths, hyperparameters, and other settings for the training process.

In the `configs/` sub-folder, you can find four pre-defined configuration files that were used to train the binary classification models for four clinical diagnostic categories: 
   - `0_control.json`
   - `1_pneumonia.json`
   - `2_wheezing.json`
   - `3_bronchiolitis.json`

To use one of these configuration files, simply pass the file path as an argument when running the `train_audio_classifier.py` script. For example, to train a model that discriminates between samples of the wheezing disorder class and samples of the remaining three classes, you can run the following command from the command line:

```
python train_audio_classifier.py configs/2_wheezing.json
```

This will start the training process using the specified configuration file. You can modify the configuration files or create your own to customize the training process for your own needs.

To compute predictions based on the first (2.5, 5.0, ..., 30.0) seconds of your recordings, you can run the `duration_experiments.py` script.
It uses the same input arguments as `train_audio_classifier.py`.
  
## License

The code in this repository is released under the Apache-2.0 license. See the `LICENSE` file for more details.

## Citation

If you use this code in your research, please cite the following paper:

```
@Article{Heitmann2023,
  author  = {Heitmann, Julien and Glangetas, Alban and Doenz, Jonathan and Dervaux, Juliane and Shama, Deeksha M. and Garcia, Daniel Hinjos and Benissa, Mohamed Rida and Cantais, Aymeric and Perez, Alexandre and M{\"u}ller, Daniel and Chavdarova, Tatjana and Ruchonnet-Metrailler, Isabelle and Siebert, Johan N. and Lacroix, Laurence and Jaggi, Martin and Gervaix, Alain and Hartley, Mary-Anne and Hugon, Florence and Fassbind, Derrick and Barro, Makura and Bediang, Georges and Hafidi, N. E. L. and Bouskraoui, M. and Ba, Idrissa and with the Pneumoscope Study Group},
  journal = {npj Digital Medicine},
  title   = {DeepBreath---automated detection of respiratory pathology from lung auscultation in 572 pediatric outpatients across 5 countries},
  year    = {2023},
  volume  = {6},
  number  = {1},
  pages   = {104},
  doi     = {10.1038/s41746-023-00838-3}
}
```
