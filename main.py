
import librosa
import os
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# ==================================================
# 1. DATASET PATHS
# ==================================================

original_dataset_path = r"D:\Baby_Cry_Project\Baby Cry Set"

external_dataset_path = r"D:\Baby_Cry_Project\External Dataset\Baby Cry Dataset"


# ==================================================
# 2. CLASS NAMES
# ==================================================

classes = [
    "belly pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "laugh",
    "noise",
    "silence",
    "tired"
]


# ==================================================
# 3. SUPPORTED AUDIO FORMATS
# ==================================================

supported_extensions = [
    ".wav",
    ".ogg",
    ".mp3"
]


# ==================================================
# 4. FEATURE EXTRACTION
# ==================================================

def extract_features(audio, sample_rate):

    # Make audio exactly 6 seconds

    target_length = 16000 * 6


    if len(audio) < target_length:

        padding = target_length - len(audio)

        audio = np.pad(
            audio,
            (0, padding)
        )

    else:

        audio = audio[:target_length]


    # Divide audio into 3 parts

    parts = np.array_split(
        audio,
        3
    )


    all_features = []


    for part in parts:

        # ------------------------------------------
        # MFCC
        # ------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=part,
            sr=sample_rate,
            n_mfcc=40
        )


        mfcc_mean = np.mean(
            mfcc,
            axis=1
        )


        mfcc_std = np.std(
            mfcc,
            axis=1
        )


        # ------------------------------------------
        # MFCC DELTA
        # ------------------------------------------

        delta = librosa.feature.delta(
            mfcc
        )


        delta_mean = np.mean(
            delta,
            axis=1
        )


        # ------------------------------------------
        # SPECTRAL CENTROID
        # ------------------------------------------

        spectral_centroid = librosa.feature.spectral_centroid(
            y=part,
            sr=sample_rate
        )


        spectral_centroid_mean = np.mean(
            spectral_centroid
        )


        # ------------------------------------------
        # ZERO CROSSING RATE
        # ------------------------------------------

        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            part
        )


        zero_crossing_rate_mean = np.mean(
            zero_crossing_rate
        )


        # ------------------------------------------
        # COMBINE FEATURES
        # ------------------------------------------

        part_features = np.concatenate(
            (
                mfcc_mean,
                mfcc_std,
                delta_mean,
                [spectral_centroid_mean],
                [zero_crossing_rate_mean]
            )
        )


        all_features.extend(
            part_features
        )


    return np.array(
        all_features
    )


# ==================================================
# 5. LOAD DATASET
# ==================================================

X = []
y = []


dataset_paths = [
    original_dataset_path,
    external_dataset_path
]


for dataset_path in dataset_paths:

    print("\nReading dataset:")
    print(dataset_path)


    if not os.path.exists(dataset_path):

        print(
            "Dataset path not found:",
            dataset_path
        )

        continue


    for class_name in classes:

        class_path = os.path.join(
            dataset_path,
            class_name
        )


        if not os.path.isdir(class_path):

            print(
                "Class folder not found:",
                class_path
            )

            continue


        for file_name in sorted(
            os.listdir(class_path)
        ):

            file_extension = os.path.splitext(
                file_name
            )[1].lower()


            if file_extension not in supported_extensions:

                continue


            file_path = os.path.join(
                class_path,
                file_name
            )


            try:

                # Load audio and resample to 16 kHz

                audio, sample_rate = librosa.load(
                    file_path,
                    sr=16000,
                    mono=True
                )


                # Extract features

                features = extract_features(
                    audio,
                    sample_rate
                )


                # Store features and labels

                X.append(
                    features
                )

                y.append(
                    class_name
                )


            except Exception as e:

                print(
                    "\nError reading:",
                    file_path
                )

                print(
                    e
                )


# ==================================================
# 6. CONVERT TO NUMPY ARRAYS
# ==================================================

X = np.array(
    X
)


y = np.array(
    y
)


# ==================================================
# 7. DATASET INFORMATION
# ==================================================

print("\n")
print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)


print(
    "\nTotal audio files:",
    len(X)
)


print(
    "Feature matrix shape:",
    X.shape
)


print(
    "Labels shape:",
    y.shape
)


if len(X) == 0:

    print(
        "\nNo audio files were processed."
    )

    print(
        "Check your dataset paths."
    )

    raise SystemExit


print(
    "Features per audio:",
    X.shape[1]
)


print(
    "First label:",
    y[0]
)


# ==================================================
# 8. CLASS COUNTS
# ==================================================

print("\n")
print("=" * 50)
print("CLASS COUNTS")
print("=" * 50)


for class_name in classes:

    count = np.sum(
        y == class_name
    )


    print(
        class_name,
        ":",
        count
    )


# ==================================================
# 9. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n")
print("=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)


print(
    "\nTraining data:",
    X_train.shape
)


print(
    "Testing data:",
    X_test.shape
)


print(
    "Training labels:",
    y_train.shape
)


print(
    "Testing labels:",
    y_test.shape
)


# ==================================================
# 10. SVM MODEL
# ==================================================

print("\n")
print("=" * 50)
print("SVM MODEL")
print("=" * 50)


# Scale features

svm_scaler = StandardScaler()


X_train_scaled = svm_scaler.fit_transform(
    X_train
)


X_test_scaled = svm_scaler.transform(
    X_test
)


# Use balanced class weights

svm_model = SVC(

    kernel="rbf",

    C=10,

    gamma="scale",

    class_weight="balanced",

    random_state=42
)


svm_model.fit(
    X_train_scaled,
    y_train
)


print(
    "\nSVM training completed!"
)


svm_pred = svm_model.predict(
    X_test_scaled
)


print(
    "SVM predictions completed!"
)


svm_accuracy = accuracy_score(
    y_test,
    svm_pred
)


print(
    "\nSVM Accuracy:",
    svm_accuracy
)


print(
    "SVM Accuracy (%):",
    svm_accuracy * 100
)


print(
    "\nSVM Classification Report:"
)


print(
    classification_report(
        y_test,
        svm_pred,
        labels=classes,
        zero_division=0
    )
)


print(
    "\nSVM Confusion Matrix:"
)


print(
    confusion_matrix(
        y_test,
        svm_pred,
        labels=classes
    )
)


# ==================================================
# 11. RANDOM FOREST MODEL
# ==================================================

print("\n")
print("=" * 50)
print("RANDOM FOREST MODEL")
print("=" * 50)


random_forest_model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)


random_forest_model.fit(
    X_train,
    y_train
)


print(
    "\nRandom Forest training completed!"
)


random_forest_pred = random_forest_model.predict(
    X_test
)


print(
    "Random Forest predictions completed!"
)


random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)


print(
    "\nRandom Forest Accuracy:",
    random_forest_accuracy
)


print(
    "Random Forest Accuracy (%):",
    random_forest_accuracy * 100
)


print(
    "\nRandom Forest Classification Report:"
)


print(
    classification_report(
        y_test,
        random_forest_pred,
        labels=classes,
        zero_division=0
    )
)


print(
    "\nRandom Forest Confusion Matrix:"
)


print(
    confusion_matrix(
        y_test,
        random_forest_pred,
        labels=classes
    )
)


# ==================================================
# 12. EXTRA TREES MODEL
# ==================================================

print("\n")
print("=" * 50)
print("EXTRA TREES MODEL")
print("=" * 50)


extra_trees_model = ExtraTreesClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)


extra_trees_model.fit(
    X_train,
    y_train
)


print(
    "\nExtra Trees training completed!"
)


extra_trees_pred = extra_trees_model.predict(
    X_test
)


print(
    "Extra Trees predictions completed!"
)


extra_trees_accuracy = accuracy_score(
    y_test,
    extra_trees_pred
)


print(
    "\nExtra Trees Accuracy:",
    extra_trees_accuracy
)


print(
    "Extra Trees Accuracy (%):",
    extra_trees_accuracy * 100
)


print(
    "\nExtra Trees Classification Report:"
)


print(
    classification_report(
        y_test,
        extra_trees_pred,
        labels=classes,
        zero_division=0
    )
)


print(
    "\nExtra Trees Confusion Matrix:"
)


print(
    confusion_matrix(
        y_test,
        extra_trees_pred,
        labels=classes
    )
)


# ==================================================
# 13. PCA VISUALIZATION
# ==================================================

print("\n")
print("=" * 50)
print("PCA VISUALIZATION")
print("=" * 50)


pca_scaler = StandardScaler()


X_scaled = pca_scaler.fit_transform(
    X
)


pca = PCA(
    n_components=2
)


X_pca = pca.fit_transform(
    X_scaled
)


print(
    "\nPCA shape:",
    X_pca.shape
)


print(
    "PCA explained variance:",
    pca.explained_variance_ratio_
)


print(
    "Total explained variance:",
    np.sum(
        pca.explained_variance_ratio_
    )
)


plt.figure(
    figsize=(10, 7)
)


for class_name in classes:

    mask = y == class_name


    if np.sum(mask) == 0:

        continue


    plt.scatter(

        X_pca[mask, 0],

        X_pca[mask, 1],

        label=class_name,

        alpha=0.6
    )


plt.xlabel(
    "PCA Component 1"
)


plt.ylabel(
    "PCA Component 2"
)


plt.title(
    "Combined Baby Cry Dataset - PCA Visualization"
)


plt.legend()


plt.tight_layout()


plt.show()


# ==================================================
# 14. FINAL ACCURACY SUMMARY
# ==================================================

print("\n")
print("=" * 50)
print("FINAL ACCURACY SUMMARY")
print("=" * 50)


print(
    "\nSVM Accuracy (%):",
    round(svm_accuracy * 100, 2)
)


print(
    "Random Forest Accuracy (%):",
    round(random_forest_accuracy * 100, 2)
)


print(
    "Extra Trees Accuracy (%):",
    round(extra_trees_accuracy * 100, 2)
)


print(
    "\nProgram completed successfully!"
)