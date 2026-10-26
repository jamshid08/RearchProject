"""
extract_features.py

Loads DEAP preprocessed subject files (s01.dat ... s32.dat), extracts
Differential Entropy (DE) features over sliding windows for 5 frequency
bands across 32 EEG channels, and saves per-subject feature/label arrays.

For each subject:
    X_subject_XX.npy -> shape (40, 160)   (40 trials, 32 channels * 5 bands)
    Y_subject_XX.npy -> shape (40, 4)     (Valence, Arousal, Dominance, Liking)
"""

import os
import pickle

import numpy as np
import scipy
from scipy import signal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FOLDER = r"C:\Users\student\Desktop\practise DEAP dataset\data_preprocessed_python"
OUTPUT_FOLDER = DATA_FOLDER  # feature files are saved alongside the .dat files

N_CHANNELS = 32          # first 32 channels of the DEAP 'data' array are EEG
FS = 128                 # DEAP preprocessed sampling rate (Hz)

WINDOW_SIZE = 4 * FS     # 4-second window  -> 512 samples
STEP_SIZE = 2 * FS       # 2-second step    -> 256 samples
NPERSEG = 256            # welch segment length

FREQ_BANDS = {
    "delta": (1, 3),
    "theta": (4, 7),
    "alpha": (8, 13),
    "beta": (14, 30),
    "gamma": (31, 45),
}


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def differential_entropy_from_power(power):
    """DE = 0.5 * log(2 * pi * e * power)."""
    power = max(power, 1e-12)  # avoid log(0)
    return 0.5 * np.log(2 * np.pi * np.e * power)


def compute_de_features_for_window(window_data):
    """
    window_data: array of shape (n_channels, window_size)
    Returns a flat vector of length n_channels * n_bands, ordered
    channel-major (ch0: delta,theta,alpha,beta,gamma, ch1: ..., ...).
    """
    n_channels = window_data.shape[0]
    features = np.zeros(n_channels * len(FREQ_BANDS))

    idx = 0
    for ch in range(n_channels):
        freqs, psd = signal.welch(window_data[ch], fs=FS, nperseg=NPERSEG)
        for band_name, (low, high) in FREQ_BANDS.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.mean(psd[band_mask]) if np.any(band_mask) else 1e-12
            features[idx] = differential_entropy_from_power(band_power)
            idx += 1

    return features


def extract_features_for_subject(file_path):
    """
    Loads a single subject .dat file and returns (X, Y):
        X -> (40, 160) DE feature matrix
        Y -> (40, 4)   [Valence, Arousal, Dominance, Liking] ratings
    """
    with open(file_path, "rb") as f:
        subject_dict = pickle.load(f, encoding="latin1")

    data = subject_dict["data"]      # (40 trials, channels, timepoints)
    labels = subject_dict["labels"]  # (40 trials, 4)

    n_trials = data.shape[0]
    eeg_data = data[:, :N_CHANNELS, :]  # keep only the 32 EEG channels

    n_features_per_window = N_CHANNELS * len(FREQ_BANDS)
    X = np.zeros((n_trials, n_features_per_window))
    Y = np.array(labels[:, :4])

    for trial_idx in range(n_trials):
        trial_signal = eeg_data[trial_idx]        # (32, timepoints)
        n_timepoints = trial_signal.shape[1]

        window_features = []
        start = 0
        while start + WINDOW_SIZE <= n_timepoints:
            window = trial_signal[:, start:start + WINDOW_SIZE]
            window_features.append(compute_de_features_for_window(window))
            start += STEP_SIZE

        if not window_features:
            # trial shorter than one window: fall back to using it whole
            window_features.append(compute_de_features_for_window(trial_signal))

        X[trial_idx] = np.mean(window_features, axis=0)

    return X, Y


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    subject_files = sorted(
        f for f in os.listdir(DATA_FOLDER)
        if f.lower().startswith("s") and f.lower().endswith(".dat")
    )

    if not subject_files:
        print(f"No .dat files found in {DATA_FOLDER}")
        return

    for fname in subject_files:
        subject_id = os.path.splitext(fname)[0]   # e.g. "s01"
        subject_num = subject_id[1:]               # e.g. "01"
        file_path = os.path.join(DATA_FOLDER, fname)

        print(f"Processing subject {subject_id}...")
        X, Y = extract_features_for_subject(file_path)

        x_out = os.path.join(OUTPUT_FOLDER, f"X_subject_{subject_num}.npy")
        y_out = os.path.join(OUTPUT_FOLDER, f"Y_subject_{subject_num}.npy")
        np.save(x_out, X)
        np.save(y_out, Y)

        print(f"  -> X shape {X.shape}, Y shape {Y.shape} saved.")

    print("Done. All subjects processed.")


if __name__ == "__main__":
    main()