import os
import pickle
import numpy as np

# ============================================================
# 1. SET THE CORRECT FOLDER PATH (FIXED - No filename at the end)
# ============================================================
# IMPORTANT: Point this to the FOLDER containing your .dat files
folder_path = r'C:\Users\student\Desktop\practise DEAP dataset\data_preprocessed_python'

# ============================================================
# 2. AUTO-FIND YOUR DATA FILE
# ============================================================
found_files = []
for file in os.listdir(folder_path):
    if file.endswith(('.dat', '.pkl')):
        found_files.append(file)

if not found_files:
    print(f"❌ No .dat or .pkl files found in: {folder_path}")
    print("   Please check if the folder path is correct.")
    exit()

file_path = os.path.join(folder_path, found_files[0])
print(f"📂 Loading file: {found_files[0]}\n")

# ============================================================
# 2. LOAD THE DATA
# ============================================================
with open(file_path, 'rb') as f:
    deap_dict = pickle.load(f, encoding='latin1')

data = deap_dict['data']
labels = deap_dict['labels']

# ============================================================
# 3. HOW THE ARRAY IS STRUCTURED (The Memory Layout)
# ============================================================
print("=" * 70)
print("HOW THE RAW EEG ARRAY IS STORED")
print("=" * 70)
print(f"Array Shape (Dimensions): {data.shape}")
print(f"Meaning of each dimension: (Trials, Channels, Timepoints)")
print(f"  - Trials: 40 music videos watched")
print(f"  - Channels: {data.shape[1]} sensors (EEG + maybe peripheral)")
print(f"  - Timepoints: {data.shape[2]} samples (128 per second)")
print("-" * 70)

# ============================================================
# 4. THE ACTUAL NUMBERS (RAW ARRAY CONTENT)
# ============================================================
print("\n🔢 ACTUAL NUMBERS STORED IN THE ARRAY")
print("-" * 70)

# Let's look at Trial 0 (first video), Channel 0 (first EEG sensor)
trial_idx = 0
channel_idx = 0

# ---- 4a. Print the first 30 timepoints of that specific channel ----
print(f"\n1. TRIAL {trial_idx}, CHANNEL {channel_idx} - First 30 timepoints:")
raw_slice = data[trial_idx, channel_idx, 0:30]
print(f"   Values: {raw_slice}")
print(f"   Data Type (dtype): {data.dtype}")

# If it's int16, explain the scaling
if data.dtype == 'int16':
    print("   ⚠️ NOTE: Stored as 16-bit integers to save space.")
    print("      These are scaled microvolts. Usually you divide by 1e-6 to get Volts.")
elif data.dtype == 'float64':
    print("   ⚠️ NOTE: Stored as 64-bit floats (actual decimal voltage values).")

# ---- 4b. Print all channels at a SINGLE point in time ----
print(f"\n2. ALL {data.shape[1]} CHANNELS at TIME POINT 0 (First sample):")
all_channels_at_time0 = data[trial_idx, :, 0]  # All channels, first time sample
print(f"   Values: {all_channels_at_time0}")

# ---- 4c. Print a 2D slice (Channels x Time) for the first second ----
print(f"\n3. 2D SLICE (Channels 0-4 x Time 0-19) - First 5 channels, first 20 samples:")
slice_2d = data[trial_idx, 0:5, 0:20]  # 5 channels, 20 timepoints
print(f"   Shape of this slice: {slice_2d.shape}")
print("   (Rows = Channels, Columns = Time samples)")
print(f"   Values:\n{slice_2d}")

# ---- 4d. Show how to manually access a specific sample ----
print("\n4. HOW TO MANUALLY ACCESS ANY VALUE:")
print(f"   data[0, 0, 0] = {data[0, 0, 0]}  (Trial 1, Channel 1, 1st sample)")
print(f"   data[0, 0, 1] = {data[0, 0, 1]}  (Trial 1, Channel 1, 2nd sample)")
print(f"   data[0, 1, 0] = {data[0, 1, 0]}  (Trial 1, Channel 2, 1st sample)")

# ============================================================
# 5. MEMORY STORAGE (How it physically sits in RAM)
# ============================================================
print("\n" + "=" * 70)
print("PHYSICAL MEMORY LAYOUT")
print("=" * 70)
print(f"Total number of numbers stored: {data.size:,}")
print(f"Memory used by this array: {data.nbytes / (1024*1024):.2f} MB")
print(f"NumPy Memory Layout (C-contiguous/Row-major): {data.flags.c_contiguous}")
print("👉 C-contiguous means the last dimension (Timepoints) changes fastest.")
print("   So data[0,0,0], data[0,0,1], data[0,0,2] are stored right next to each other in RAM.")

# ============================================================
# 6. SUBJECTIVE RATINGS (Labels)
# ============================================================
print("\n" + "=" * 70)
print("HOW LABELS (RATINGS) ARE STORED")
print("=" * 70)
print(f"Labels Array Shape: {labels.shape}")
print(f"Ratings for Trial 0 (first video): {labels[0]}")
print("   Format: [Valence, Arousal, Dominance, Liking] (1 = lowest, 9 = highest)")