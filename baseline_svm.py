"""
baseline_svm.py

Trains a baseline (no augmentation) RBF-SVM per subject on DEAP DE features
to classify Valence and Arousal (binarized at threshold 5), and reports
per-subject and aggregate (mean/std) accuracy, precision, recall, F1.

Expects, in the current working directory:
    X_subject_01.npy ... X_subject_10.npy   (each shape: (40, 160))
    Y_subject_01.npy ... Y_subject_10.npy   (each shape: (40, 4))

Saves all results (per-subject + aggregate) to 'baseline_results.npy'.
"""

import os

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder containing X_subject_XX.npy / Y_subject_XX.npy (same folder
# extract_features.py saved them into). Update this if your files live
# somewhere else.
DATA_FOLDER = r"C:\Users\JAMSHID KHALIQ\Desktop\Project-EEGResearch\JAMSHID\practise DEAP dataset\data_preprocessed_python"

SUBJECTS = range(1, 11)          # s01 to s10
LABEL_TASKS = ["valence", "arousal"]
LABEL_COLUMN = {"valence": 0, "arousal": 1}
THRESHOLD = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def binarize_labels(Y):
    """Returns (valence_binary, arousal_binary) each shape (n_trials,)."""
    valence = (Y[:, LABEL_COLUMN["valence"]] > THRESHOLD).astype(int)
    arousal = (Y[:, LABEL_COLUMN["arousal"]] > THRESHOLD).astype(int)
    return valence, arousal


def train_and_evaluate(X, y):
    """
    Splits, scales, trains an RBF SVM, and returns a dict of metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="binary", zero_division=0),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    per_subject_results = {}  # {subject_id: {"valence": {...}, "arousal": {...}}}

    for subj in SUBJECTS:
        subj_id = f"{subj:02d}"
        X = np.load(os.path.join(DATA_FOLDER, f"X_subject_{subj_id}.npy"))
        Y = np.load(os.path.join(DATA_FOLDER, f"Y_subject_{subj_id}.npy"))

        valence_labels, arousal_labels = binarize_labels(Y)
        label_arrays = {"valence": valence_labels, "arousal": arousal_labels}

        subj_results = {}
        for task in LABEL_TASKS:
            metrics = train_and_evaluate(X, label_arrays[task])
            subj_results[task] = metrics

        per_subject_results[f"s{subj_id}"] = subj_results

    # -----------------------------------------------------------------
    # Aggregate mean/std across subjects, per task
    # -----------------------------------------------------------------
    aggregate_results = {}
    for task in LABEL_TASKS:
        task_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for subj_id in per_subject_results:
            for metric_name in task_metrics:
                task_metrics[metric_name].append(per_subject_results[subj_id][task][metric_name])

        aggregate_results[task] = {
            metric_name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for metric_name, values in task_metrics.items()
        }

    # -----------------------------------------------------------------
    # Save everything
    # -----------------------------------------------------------------
    results_to_save = {
        "per_subject": per_subject_results,
        "aggregate": aggregate_results,
    }
    results_path = os.path.join(DATA_FOLDER, "baseline_results.npy")
    np.save(results_path, results_to_save, allow_pickle=True)

    # -----------------------------------------------------------------
    # Print nicely formatted output
    # -----------------------------------------------------------------
    print_results(per_subject_results, aggregate_results, results_path)


def print_results(per_subject_results, aggregate_results, results_path):
    for task in LABEL_TASKS:
        print("\n" + "=" * 70)
        print(f"  {task.upper()} - Per-Subject Results")
        print("=" * 70)
        header = f"{'Subject':<10}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-score':<12}"
        print(header)
        print("-" * 70)
        for subj_id, results in per_subject_results.items():
            m = results[task]
            print(
                f"{subj_id:<10}"
                f"{m['accuracy']:<12.4f}"
                f"{m['precision']:<12.4f}"
                f"{m['recall']:<12.4f}"
                f"{m['f1']:<12.4f}"
            )
        print("-" * 70)
        agg = aggregate_results[task]
        print(
            f"{'Mean':<10}"
            f"{agg['accuracy']['mean']:<12.4f}"
            f"{agg['precision']['mean']:<12.4f}"
            f"{agg['recall']['mean']:<12.4f}"
            f"{agg['f1']['mean']:<12.4f}"
        )
        print(
            f"{'Std':<10}"
            f"{agg['accuracy']['std']:<12.4f}"
            f"{agg['precision']['std']:<12.4f}"
            f"{agg['recall']['std']:<12.4f}"
            f"{agg['f1']['std']:<12.4f}"
        )

    print("\n" + "=" * 70)
    print("  SUMMARY (Mean +/- Std across 10 subjects)")
    print("=" * 70)
    for task in LABEL_TASKS:
        agg = aggregate_results[task]
        print(f"\n{task.capitalize()}:")
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            mean_val = agg[metric_name]["mean"]
            std_val = agg[metric_name]["std"]
            print(f"  {metric_name.capitalize():<10}: {mean_val:.4f} +/- {std_val:.4f}")

    print(f"\nResults saved to '{results_path}'")


if __name__ == "__main__":
    main()