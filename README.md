# Systematic Comparison of Data Augmentation for Subject-Dependent EEG Emotion Recognition on DEAP

## 📖 Overview
This repository contains the official code and supplementary materials for the Master's thesis: **"Systematic Comparison of Data Augmentation for Subject-Dependent EEG Emotion Recognition on DEAP"** by Jamshid Khaliq (Central University of Kashmir).

**The Problem:** EEG-based emotion recognition suffers from a severe data scarcity problem. The widely used DEAP dataset contains only 40 trials per subject. While data augmentation is often proposed as a solution, the literature presents contradictory findings—some studies report huge gains using complex GANs, while others report zero benefit.

**The Gap:** Many prior studies suffer from the "Accuracy Trap." They fail to use class-weighted baselines, meaning their models simply predict the majority class. This makes synthetic augmentations like SMOTE appear falsely beneficial.

**Our Contribution:** We provide the first honest, systematic benchmark of five classical data augmentation techniques using a properly class-weighted Support Vector Machine (SVM) baseline. 

## 🔬 Key Findings
1. **SMOTE is Redundant:** When the baseline SVM uses `class_weight='balanced'`, SMOTE actually *degrades* performance (Valence F1 drops from 29.36% to 25.55%).
2. **Channel Dropout Wins:** For Valence, **Channel Dropout at 4× ratio** is the only technique that improves F1-score (up to **34.16%**, a +4.80% gain), acting as a superior regularizer.
3. **Subject Variability:** Augmentation effects vary wildly. Channel Dropout improved Subject s10's accuracy by +37.5%, but degraded Subject s02's by -25.0%.

## 📊 Dataset
* **Source:** [DEAP Dataset on Kaggle](https://www.kaggle.com/datasets/manh123df/deap-dataset)
* **Subjects:** 10 subjects (s01 to s10)
* **Signals:** 32-channel EEG, preprocessed (downsampled to 128 Hz, bandpass filtered 4-45 Hz).
* **Features:** Differential Entropy (DE) across 5 bands (Delta, Theta, Alpha, Beta, Gamma), resulting in a 160-dimensional vector per trial.

## ⚙️ Methodology Pipeline
1. **Feature Extraction:** 4-second sliding windows (2-second step), Welch's PSD, Differential Entropy.
2. **Label Binarization:** Valence and Arousal scores > 5.0 are labeled "High" (1), otherwise "Low" (0).
3. **Augmentation:** Applied *only* to the training set (80%) at 2× and 4× ratios.
   * Gaussian Noise
   * Channel Dropout (20% features zeroed)
   * Band Masking (1-2 bands zeroed)
   * SMOTE
   * Mixup
4. **Classifier:** SVM with RBF kernel (\(C=1.0\), \(\gamma=\text{'scale'}\), **`class_weight='balanced'`**).
5. **Evaluation:** Stratified 80/20 train-test split. Metrics: Accuracy, Precision, Recall, F1-Score.

## 📂 Repository Structure
```text
├── data_preprocessed_python/      # Folder containing DEAP .dat files (s01.dat to s10.dat)
├── plots/                         # Output folder for generated 300-DPI figures
├── extract_features.py            # Script to extract 160-D DE features from raw .dat files
├── baseline_svm.py                # Script to run the class-weighted baseline SVM
├── augmentation_svm.py            # Main script to run all 5 augmentations
├── generate_plots.py              # Script to generate heatmaps, bar charts, and line plots
└── README.md                      # Project documentation
