import numpy as np
import pickle
import os
from scipy.signal import welch

# ==========================================
# 1. CONSTANTS (Do not change these)
# ==========================================
SAMPLE_RATE = 128          # DEAP data is sampled at 128 Hz
WINDOW_SEC = 4             # Length of each window in seconds
WINDOW_SAMPLES = WINDOW_SEC * SAMPLE_RATE  # 4 * 128 = 512 data points
STEP_SEC = 2               # Step size (overlap of 2 seconds -> 50% overlap)
STEP_SAMPLES = STEP_SEC * SAMPLE_RATE      # 2 * 128 = 256 data points

# Frequency bands for brainwaves (in Hz)
BANDS = {
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45)
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def compute_band_power(psd, freqs, low, high):
    """
    Calculates the total power (area under the curve) in a specific frequency band.
    psd: Power Spectral Density array
    freqs: Frequency array corresponding to psd
    low, high: frequency range for the band (e.g., 8 to 13 Hz for Alpha)
    """
    # Find the indices where the frequency falls between 'low' and 'high'
    idx = np.logical_and(freqs >= low, freqs <= high)
    # Use trapezoidal integration to calculate the area under the PSD curve
    return np.trapezoid(psd[idx], freqs[idx])  # ✅ NEW (works in NumPy 2.0+)

def extract_features_from_trial(trial_data):
    """
    This is the core feature extraction function.
    
    Input: 
        trial_data : shape [32 channels, 7680 time points] (60 seconds of EEG)
    
    Output:
        features : shape [num_windows, 256 features]
                  Each window has 32 channels * 4 bands * 2 features = 256 features.
                  The 2 features are: [Band Power, Differential Entropy] for each band.
    """
    num_channels, total_samples = trial_data.shape
    all_windows = []  # This will store the feature vectors for every window

    # --- Step A: Sliding Window Loop ---
    # We slide from start=0 to end=(total - window_size), moving by STEP_SAMPLES each time
    for start in range(0, total_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        end = start + WINDOW_SAMPLES
        window_data = trial_data[:, start:end]  # Shape: [32, 512]

        window_features = []  # Features for this single window

        # --- Step B: Loop through each of the 32 EEG channels ---
        for ch in range(num_channels):
            signal = window_data[ch, :]  # Shape: [512] (4 seconds of data for 1 channel)

            # --- Step C: Convert Time -> Frequency (FFT / PSD) ---
            # Welch's method computes the Power Spectral Density (PSD).
            # We use nperseg=256 (2 seconds) which gives a smooth spectrum.
            freqs, psd = welch(signal, fs=SAMPLE_RATE, nperseg=min(256, WINDOW_SAMPLES))

            # --- Step D: Extract Power & Differential Entropy for each band ---
            for band_name, (low, high) in BANDS.items():
                
                # 1. BAND POWER: Calculate the total power in this frequency range
                power = compute_band_power(psd, freqs, low, high)  # ✅ CORRECT
                window_features.append(power)  # Append as Feature 1

                # 2. DIFFERENTIAL ENTROPY (DE):
                # The formula for DE for a Gaussian distribution is:
                # DE = 0.5 * ln(2 * pi * e * variance)
                # Since the EEG signal is zero-mean, variance ≈ Power.
                # So we replace variance with the band power we just calculated.
                de = 0.5 * np.log(2 * np.pi * np.e * power)
                window_features.append(de)     # Append as Feature 2

        # Add this window's features to the master list
        all_windows.append(window_features)

    # Convert list to a numpy array and return it
    return np.array(all_windows)


# ==========================================
# 3. MAIN EXECUTION (Load & Run)
# ==========================================

if __name__ == "__main__":
    
    # --- Step 1: Load the dataset ---
    # Make sure the 'data_preprocessed_python' folder is in your current directory
    data_folder = "data_preprocessed_python"
    file_path = os.path.join(data_folder, "s01.dat")  # Load Subject 1

    print(f"Loading file: {file_path}")
    
    # DEAP .dat files are saved as Python pickles.
    # 'latin1' encoding is required for Python 3 to load Python 2 pickles.
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')

    # --- Step 2: Extract EEG and Ratings ---
    # data['data'] : shape (40 trials, 32 channels, 7680 timepoints)
    # data['labels'] : shape (40 trials, 4) -> [Valence, Arousal, Dominance, Liking]
    eeg_signals = data_dict['data']
    ratings = data_dict['labels']

    print(f"EEG Signals Shape: {eeg_signals.shape} (Trials, Channels, Timepoints)")
    print(f"Ratings Shape: {ratings.shape} (Trials, 4 Ratings)")

    # --- Step 3: Process a single trial ---
    # Let's pick the very first music video (Trial 0) for demonstration
    trial_index = 0
    trial_eeg = eeg_signals[trial_index]  # Shape: [32, 7680]
    trial_rating = ratings[trial_index]   # Shape: [4]

    print("\n" + "="*50)
    print(f"Processing Trial {trial_index + 1}")
    print(f"Self-Reported Ratings for this video:")
    print(f"  Valence (1=Neg to 9=Pos): {trial_rating[0]}")
    print(f"  Arousal (1=Calm to 9=Excited): {trial_rating[1]}")
    print(f"  Dominance: {trial_rating[2]}")
    print(f"  Liking: {trial_rating[3]}")
    print("="*50)

    # --- Step 4: Extract Features ---
    features = extract_features_from_trial(trial_eeg)

    print(f"\nExtracted Feature Matrix Shape: {features.shape}")
    print(f"  - Rows: {features.shape[0]} windows (4-second windows with 2-second overlap)")
    print(f"  - Columns: {features.shape[1]} features per window")
    print(f"    (32 channels × 4 frequency bands × 2 features [Power & DE])")

    # --- Step 5: How to pair features with labels for your model ---
    # Since the 60-second trial is split into multiple windows, we need a label
    # for EACH window. We simply duplicate the rating for all windows.
    
    # Example: Create a binary label for Arousal (High > 5, Low <= 5)
    arousal_label = 1 if trial_rating[1] > 5 else 0
    labels_for_windows = np.full(features.shape[0], arousal_label)

    print(f"\nLabel assigned to all {features.shape[0]} windows: {arousal_label} (1=High Arousal, 0=Low Arousal)")

    # --- At this point, ready to train your model! ---
    # X = features
    # y = labels_for_windows
    # You can now split X and y into train/test sets and feed them into your AI model.