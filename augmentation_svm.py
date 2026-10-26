"""
augmentation_svm.py

Compares 5 data-augmentation techniques against a no-augmentation baseline
for SVM classification of Valence and Arousal on DEAP DE features, across
subjects s01-s10.

Augmentation techniques (each tested at 2x and 4x training-set size, except
SMOTE which sets its own ratio):
    1. Gaussian noise injection
    2. Channel dropout (zero out random features)
    3. Frequency-band masking (zero out 1-2 whole bands, all channels)
    4. SMOTE (synthetic minority oversampling)
    5. Mixup (convex combinations of sample pairs)

Test data is NEVER augmented - only the training split.

Requires: numpy, scikit-learn, imbalanced-learn (`pip install imbalanced-learn`)

Expects, alongside this script (or update DATA_FOLDER below):
    X_subject_01.npy ... X_subject_10.npy   (each shape: (40, 160))
    Y_subject_01.npy ... Y_subject_10.npy   (each shape: (40, 4))

Saves all results to 'augmentation_results.npy'.
"""

import os
from collections import OrderedDict

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError(
        "imbalanced-learn is required for SMOTE. Install it with:\n"
        "    pip install imbalanced-learn"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FOLDER = r"C:\Users\JAMSHID KHALIQ\Desktop\Project-EEGResearch\JAMSHID\practise DEAP dataset\data_preprocessed_python"

SUBJECTS = range(1, 11)          # s01 to s10
LABEL_TASKS = ["valence", "arousal"]
LABEL_COLUMN = {"valence": 0, "arousal": 1}
THRESHOLD = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

N_CHANNELS = 32
N_BANDS = 5                      # delta, theta, alpha, beta, gamma
N_FEATURES = N_CHANNELS * N_BANDS  # 160

RATIOS = [2, 4]                  # 2x and 4x training-set size


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def binarize_labels(Y):
    valence = (Y[:, LABEL_COLUMN["valence"]] > THRESHOLD).astype(int)
    arousal = (Y[:, LABEL_COLUMN["arousal"]] > THRESHOLD).astype(int)
    return valence, arousal


# ---------------------------------------------------------------------------
# Augmentation functions
# All take the TRAIN split only, and return (X_augmented, y_augmented)
# which includes the original samples plus synthetic ones.
# ---------------------------------------------------------------------------

def add_gaussian_noise(X, y, noise_std=0.1, ratio=2, rng=None):
    """X_noisy = X + N(0, noise_std). Returns original + (ratio-1) noisy copies."""
    rng = rng or np.random.RandomState(RANDOM_STATE)
    X_parts, y_parts = [X], [y]
    for _ in range(ratio - 1):
        noise = rng.normal(0, noise_std, X.shape)
        X_parts.append(X + noise)
        y_parts.append(y)
    return np.vstack(X_parts), np.concatenate(y_parts)


def channel_dropout(X, y, dropout_rate=0.2, ratio=2, rng=None):
    """Randomly zeros ~dropout_rate of the 160 features per synthetic sample."""
    rng = rng or np.random.RandomState(RANDOM_STATE)
    n_drop = int(round(dropout_rate * X.shape[1]))  # e.g. 0.2 * 160 = 32
    X_parts, y_parts = [X], [y]
    for _ in range(ratio - 1):
        X_copy = X.copy()
        for i in range(X_copy.shape[0]):
            drop_idx = rng.choice(X_copy.shape[1], size=n_drop, replace=False)
            X_copy[i, drop_idx] = 0.0
        X_parts.append(X_copy)
        y_parts.append(y)
    return np.vstack(X_parts), np.concatenate(y_parts)


def band_masking(X, y, ratio=2, rng=None):
    """
    Randomly zeros out 1-2 entire frequency bands (all 32 channels) per
    synthetic sample. Features are channel-major:
        [Ch1_Delta, Ch1_Theta, Ch1_Alpha, Ch1_Beta, Ch1_Gamma, Ch2_Delta, ...]
    so band b's feature indices are every N_BANDS-th index starting at b.
    """
    rng = rng or np.random.RandomState(RANDOM_STATE)
    X_parts, y_parts = [X], [y]
    for _ in range(ratio - 1):
        X_copy = X.copy()
        for i in range(X_copy.shape[0]):
            n_bands_to_mask = rng.choice([1, 2])
            bands_to_mask = rng.choice(N_BANDS, size=n_bands_to_mask, replace=False)
            for band in bands_to_mask:
                band_idx = np.arange(band, N_FEATURES, N_BANDS)
                X_copy[i, band_idx] = 0.0
        X_parts.append(X_copy)
        y_parts.append(y)
    return np.vstack(X_parts), np.concatenate(y_parts)


def apply_smote(X, y, random_state=RANDOM_STATE):
    """
    Balances classes with SMOTE. k_neighbors is capped to the minority
    class size (SMOTE fails if k_neighbors >= minority sample count).
    Returns the original (X, y) unchanged if SMOTE cannot be applied
    (e.g. a class has fewer than 2 samples).
    """
    class_counts = np.bincount(y)
    minority_count = class_counts[class_counts > 0].min()

    if minority_count < 2 or len(class_counts[class_counts > 0]) < 2:
        # Not enough minority samples, or only one class present
        return X, y

    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def apply_mixup(X, y, alpha=0.2, ratio=2, rng=None):
    """
    Synthetic samples: X_new = lambda*X_i + (1-lambda)*X_j, lambda ~ Beta(a,a).
    Label of the new sample is kept as y_i (the first sample's label).
    Returns original + (ratio-1)*n_original mixed samples.
    """
    rng = rng or np.random.RandomState(RANDOM_STATE)
    n_original = X.shape[0]
    X_parts, y_parts = [X], [y]
    for _ in range(ratio - 1):
        X_mixed = np.zeros_like(X)
        y_mixed = np.zeros_like(y)
        for i in range(n_original):
            j = rng.randint(0, n_original)
            lam = rng.beta(alpha, alpha)
            X_mixed[i] = lam * X[i] + (1 - lam) * X[j]
            y_mixed[i] = y[i]
        X_parts.append(X_mixed)
        y_parts.append(y_mixed)
    return np.vstack(X_parts), np.concatenate(y_parts)


# ---------------------------------------------------------------------------
# Build the ordered list of techniques to evaluate
# ---------------------------------------------------------------------------

def build_technique_list():
    """
    Returns an OrderedDict: name -> callable(X_train, y_train, rng) -> (X_aug, y_aug)
    """
    techniques = OrderedDict()
    techniques["baseline"] = lambda X, y, rng: (X, y)

    for ratio in RATIOS:
        techniques[f"gaussian_noise_{ratio}x"] = (
            lambda X, y, rng, r=ratio: add_gaussian_noise(X, y, noise_std=0.1, ratio=r, rng=rng)
        )
    for ratio in RATIOS:
        techniques[f"channel_dropout_{ratio}x"] = (
            lambda X, y, rng, r=ratio: channel_dropout(X, y, dropout_rate=0.2, ratio=r, rng=rng)
        )
    for ratio in RATIOS:
        techniques[f"band_masking_{ratio}x"] = (
            lambda X, y, rng, r=ratio: band_masking(X, y, ratio=r, rng=rng)
        )

    techniques["smote"] = lambda X, y, rng: apply_smote(X, y)

    for ratio in RATIOS:
        techniques[f"mixup_{ratio}x"] = (
            lambda X, y, rng, r=ratio: apply_mixup(X, y, alpha=0.2, ratio=r, rng=rng)
        )

    return techniques

TECHNIQUES = build_technique_list()


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # THIS IS WHERE YOU ADD CLASS WEIGHT
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced") 
    
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="binary", zero_division=0),
    }


def run_subject_task(X, y, seed):
    """
    Runs the full train/test split + all techniques for one subject/task.
    Returns dict: technique_name -> metrics dict.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    results = {}
    for name, augment_fn in TECHNIQUES.items():
        rng = np.random.RandomState(seed)
        X_aug, y_aug = augment_fn(X_train, y_train, rng)
        results[name] = train_and_evaluate_svm(X_aug, y_aug, X_test, y_test)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    per_subject_results = {}  # {subj_id: {task: {technique: metrics}}}

    for subj in SUBJECTS:
        subj_id = f"{subj:02d}"
        X = np.load(os.path.join(DATA_FOLDER, f"X_subject_{subj_id}.npy"))
        Y = np.load(os.path.join(DATA_FOLDER, f"Y_subject_{subj_id}.npy"))
        valence_labels, arousal_labels = binarize_labels(Y)
        label_arrays = {"valence": valence_labels, "arousal": arousal_labels}

        print(f"Processing subject s{subj_id}...")
        subj_results = {}
        for task in LABEL_TASKS:
            seed = RANDOM_STATE + subj  # reproducible but varies per subject
            subj_results[task] = run_subject_task(X, label_arrays[task], seed)

        per_subject_results[f"s{subj_id}"] = subj_results

    aggregate_results = compute_aggregate(per_subject_results)
    improvement_results = compute_improvement(aggregate_results)

    results_to_save = {
        "per_subject": per_subject_results,
        "aggregate": aggregate_results,
        "improvement_over_baseline": improvement_results,
    }
    results_path = os.path.join(DATA_FOLDER, "augmentation_results.npy")
    np.save(results_path, results_to_save, allow_pickle=True)

    print_results(per_subject_results, aggregate_results, improvement_results, results_path)


def compute_aggregate(per_subject_results):
    """aggregate[task][technique][metric] = {mean, std} across subjects."""
    aggregate = {}
    for task in LABEL_TASKS:
        aggregate[task] = {}
        for technique in TECHNIQUES:
            metric_values = {"accuracy": [], "precision": [], "recall": [], "f1": []}
            for subj_id in per_subject_results:
                m = per_subject_results[subj_id][task][technique]
                for metric_name in metric_values:
                    metric_values[metric_name].append(m[metric_name])
            aggregate[task][technique] = {
                metric_name: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for metric_name, vals in metric_values.items()
            }
    return aggregate


def compute_improvement(aggregate_results):
    """improvement[task][technique][metric] = technique_mean - baseline_mean."""
    improvement = {}
    for task in LABEL_TASKS:
        improvement[task] = {}
        baseline_means = {
            m: aggregate_results[task]["baseline"][m]["mean"]
            for m in ["accuracy", "precision", "recall", "f1"]
        }
        for technique in TECHNIQUES:
            if technique == "baseline":
                continue
            improvement[task][technique] = {
                m: aggregate_results[task][technique][m]["mean"] - baseline_means[m]
                for m in ["accuracy", "precision", "recall", "f1"]
            }
    return improvement


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results(per_subject_results, aggregate_results, improvement_results, results_path):
    for task in LABEL_TASKS:
        print("\n" + "=" * 90)
        print(f"  {task.upper()} - Per-Subject Accuracy by Technique (Baseline vs Augmentations)")
        print("=" * 90)
        col_width = 12
        header = f"{'Subject':<10}" + "".join(f"{t:<{col_width}}" for t in TECHNIQUES)
        print(header)
        print("-" * len(header))
        for subj_id, task_results in per_subject_results.items():
            row = f"{subj_id:<10}"
            for technique in TECHNIQUES:
                acc = task_results[task][technique]["accuracy"]
                row += f"{acc:<{col_width}.4f}"
            print(row)

        print("\n  Mean +/- Std across 10 subjects (per technique):")
        print(f"  {'Technique':<20}{'Accuracy':<16}{'Precision':<16}{'Recall':<16}{'F1-score':<16}")
        print("  " + "-" * 84)
        for technique in TECHNIQUES:
            agg = aggregate_results[task][technique]
            print(
                f"  {technique:<20}"
                f"{agg['accuracy']['mean']:.4f}+/-{agg['accuracy']['std']:.4f}  "
                f"{agg['precision']['mean']:.4f}+/-{agg['precision']['std']:.4f}  "
                f"{agg['recall']['mean']:.4f}+/-{agg['recall']['std']:.4f}  "
                f"{agg['f1']['mean']:.4f}+/-{agg['f1']['std']:.4f}"
            )

        print("\n  Average improvement over baseline:")
        print(f"  {'Technique':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-score':<12}")
        print("  " + "-" * 68)
        for technique in TECHNIQUES:
            if technique == "baseline":
                continue
            imp = improvement_results[task][technique]
            print(
                f"  {technique:<20}"
                f"{imp['accuracy']:+.4f}    "
                f"{imp['precision']:+.4f}    "
                f"{imp['recall']:+.4f}    "
                f"{imp['f1']:+.4f}"
            )

    print(f"\nResults saved to '{results_path}'")


if __name__ == "__main__":
    main()