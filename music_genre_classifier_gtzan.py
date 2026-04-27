from pathlib import Path
import warnings

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from IPython.display import Audio, display
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

PROJECT_ROOT = Path.cwd()
HOME = Path.home()

MANUAL_DATASET_ROOT = None

search_base_folders = [
    PROJECT_ROOT,
    PROJECT_ROOT.parent,
    HOME / "Downloads",
    HOME / "Desktop",
    HOME / "Documents",
]

possible_dataset_roots = []

for base_folder in search_base_folders:
    possible_dataset_roots.extend(
        [
            base_folder / "Data" / "genres_original",
            base_folder / "genres_original",
            base_folder / "gtzan-dataset-music-genre-classification" / "Data" / "genres_original",
            base_folder / "archive" / "Data" / "genres_original",
            base_folder / "input" / "gtzan-dataset-music-genre-classification" / "Data" / "genres_original",
        ]
    )

if MANUAL_DATASET_ROOT is not None:
    possible_dataset_roots.insert(0, Path(MANUAL_DATASET_ROOT))

DATASET_ROOT = None

for candidate_root in possible_dataset_roots:
    if candidate_root.exists():
        DATASET_ROOT = candidate_root
        break

if DATASET_ROOT is None:
    for base_folder in search_base_folders:
        if not base_folder.exists():
            continue
        for found_folder in base_folder.rglob("genres_original"):
            if found_folder.is_dir():
                DATASET_ROOT = found_folder
                break
        if DATASET_ROOT is not None:
            break

if DATASET_ROOT is None:
    print("Searched these folders:")
    for searched_path in possible_dataset_roots:
        print(searched_path)
    raise FileNotFoundError(
        "Could not find the GTZAN genres_original folder. "
        "Put the extracted dataset beside this notebook or set MANUAL_DATASET_ROOT to your genres_original path."
    )

print(f"Dataset folder found at: {DATASET_ROOT}")

EXPECTED_GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

genre_dirs = [DATASET_ROOT / genre for genre in EXPECTED_GENRES]
missing_genres = [genre for genre, path in zip(EXPECTED_GENRES, genre_dirs) if not path.exists()]

if missing_genres:
    raise FileNotFoundError(f"Missing genre folders: {missing_genres}")

print("All 10 expected genre folders are present.")

audio_extensions = {".wav", ".au", ".mp3"}
records = []

for genre in EXPECTED_GENRES:
    genre_folder = DATASET_ROOT / genre
    for audio_path in sorted(genre_folder.iterdir()):
        if audio_path.suffix.lower() in audio_extensions:
            records.append(
                {
                    "genre": genre,
                    "filename": audio_path.name,
                    "path": audio_path,
                }
            )

file_df = pd.DataFrame(records)

print(f"Total audio files found: {len(file_df)}")
display(file_df.head())

clip_count_df = (
    file_df.groupby("genre")
    .size()
    .reset_index(name="clip_count")
    .sort_values("genre")
)

display(clip_count_df)

plt.figure(figsize=(11, 4))
sns.barplot(data=clip_count_df, x="genre", y="clip_count", palette="Set2")
plt.title("Number of Audio Clips per Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Clips")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

sample_row = file_df.iloc[0]
sample_path = sample_row["path"]
sample_genre = sample_row["genre"]

sample_audio, sample_rate = librosa.load(sample_path, sr=None, mono=True)
sample_duration = len(sample_audio) / sample_rate

print(f"Sample file: {sample_path.name}")
print(f"Genre label: {sample_genre}")
print(f"Audio array shape: {sample_audio.shape}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {sample_duration:.2f} seconds")

display(Audio(sample_audio, rate=sample_rate))

def load_audio_30_seconds(audio_path, target_sample_rate=22050):
    audio, sample_rate = librosa.load(
        audio_path,
        sr=target_sample_rate,
        mono=True,
        duration=30,
    )
    return audio, sample_rate

def plot_audio_views(audio_path, genre_name):
    audio, sample_rate = load_audio_30_seconds(audio_path)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"{genre_name.upper()} sample: {Path(audio_path).name}", fontsize=16)

    librosa.display.waveshow(audio, sr=sample_rate, ax=axes[0])
    axes[0].set_title("Waveform: amplitude/loudness over time")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")

    stft = librosa.stft(audio)
    stft_magnitude = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    spec_image = librosa.display.specshow(
        spectrogram_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="hz",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_title("Spectrogram: frequency energy over time")
    fig.colorbar(spec_image, ax=axes[1], format="%+2.0f dB")

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_image = librosa.display.specshow(
        mfcc,
        sr=sample_rate,
        x_axis="time",
        ax=axes[2],
        cmap="viridis",
    )
    axes[2].set_title("MFCC Heatmap: compact timbre pattern over time")
    axes[2].set_ylabel("MFCC coefficient")
    fig.colorbar(mfcc_image, ax=axes[2])

    plt.tight_layout()
    plt.show()

for genre_name in EXPECTED_GENRES:
    genre_files = file_df[file_df["genre"] == genre_name]
    first_file_for_genre = genre_files.iloc[0]["path"]
    plot_audio_views(first_file_for_genre, genre_name)

def add_mean_and_std_features(feature_dict, feature_matrix, feature_prefix):
    feature_means = np.mean(feature_matrix, axis=1)
    feature_stds = np.std(feature_matrix, axis=1)

    for index, value in enumerate(feature_means):
        feature_dict[f"{feature_prefix}_{index + 1}_mean"] = value

    for index, value in enumerate(feature_stds):
        feature_dict[f"{feature_prefix}_{index + 1}_std"] = value

    return feature_dict

def extract_audio_features(audio_path, target_sample_rate=22050, duration=30):
    audio, sample_rate = librosa.load(
        audio_path,
        sr=target_sample_rate,
        mono=True,
        duration=duration,
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty or could not be decoded.")

    features = {}

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=20,
    )
    features = add_mean_and_std_features(features, mfcc, "mfcc")

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate,
    )
    features = add_mean_and_std_features(features, chroma, "chroma")

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
    )
    features["spectral_centroid_mean"] = np.mean(spectral_centroid)
    features["spectral_centroid_std"] = np.std(spectral_centroid)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=audio
    )
    features["zero_crossing_rate_mean"] = np.mean(zero_crossing_rate)
    features["zero_crossing_rate_std"] = np.std(zero_crossing_rate)

    rms_energy = librosa.feature.rms(
        y=audio
    )
    features["rms_energy_mean"] = np.mean(rms_energy)
    features["rms_energy_std"] = np.std(rms_energy)

    return features

test_features = extract_audio_features(sample_path)
test_feature_df = pd.DataFrame([test_features])

print(f"Number of features extracted from one clip: {test_feature_df.shape[1]}")
display(test_feature_df.head())

feature_rows = []
failed_files = []
total_files = len(file_df)

for index, row in file_df.reset_index(drop=True).iterrows():
    audio_path = row["path"]
    genre_name = row["genre"]

    try:
        extracted_features = extract_audio_features(audio_path)
        extracted_features["genre"] = genre_name
        extracted_features["filename"] = audio_path.name
        extracted_features["file_path"] = str(audio_path)
        feature_rows.append(extracted_features)
    except Exception as error:
        failed_files.append({"path": str(audio_path), "error": str(error)})

    if (index + 1) % 50 == 0 or (index + 1) == total_files:
        print(f"Processed {index + 1}/{total_files} files")

features_df = pd.DataFrame(feature_rows)
failed_df = pd.DataFrame(failed_files)

print(f"Feature rows created: {len(features_df)}")
print(f"Files failed: {len(failed_df)}")

if not failed_df.empty:
    display(failed_df)

display(features_df.head())

output_csv_path = PROJECT_ROOT / "gtzan_extracted_audio_features.csv"
features_df.to_csv(output_csv_path, index=False)

print(f"Saved extracted features to: {output_csv_path}")
print(f"Feature matrix shape: {features_df.shape}")

metadata_columns = ["genre", "filename", "file_path"]
feature_columns = [column for column in features_df.columns if column not in metadata_columns]

X = features_df[feature_columns]
y = features_df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training rows: {X_train.shape[0]}")
print(f"Testing rows: {X_test.shape[0]}")
print(f"Number of input features: {X_train.shape[1]}")

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=RANDOM_STATE,
)

svm_model.fit(X_train_scaled, y_train)

print("SVM training complete.")

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_model.fit(X_train_scaled, y_train)

print("Random Forest training complete.")

def evaluate_classifier(model_name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)
    accuracy = accuracy_score(y_eval, predictions)
    report = classification_report(
        y_eval,
        predictions,
        labels=EXPECTED_GENRES,
        zero_division=0,
    )

    return {
        "model_name": model_name,
        "predictions": predictions,
        "accuracy": accuracy,
        "report": report,
    }

svm_results = evaluate_classifier("SVM", svm_model, X_test_scaled, y_test)
rf_results = evaluate_classifier("Random Forest", rf_model, X_test_scaled, y_test)

comparison_df = pd.DataFrame(
    [
        {"Model": svm_results["model_name"], "Accuracy": svm_results["accuracy"]},
        {"Model": rf_results["model_name"], "Accuracy": rf_results["accuracy"]},
    ]
)

display(comparison_df)

print("SVM Classification Report")
print(svm_results["report"])

print("Random Forest Classification Report")
print(rf_results["report"])

def plot_confusion_matrix_on_axis(axis, y_true, y_pred, title):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=EXPECTED_GENRES,
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EXPECTED_GENRES,
        yticklabels=EXPECTED_GENRES,
        ax=axis,
    )

    axis.set_title(title)
    axis.set_xlabel("Predicted Genre")
    axis.set_ylabel("True Genre")

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
plot_confusion_matrix_on_axis(axes[0], y_test, svm_results["predictions"], "SVM Confusion Matrix")
plot_confusion_matrix_on_axis(axes[1], y_test, rf_results["predictions"], "Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

def predict_custom_audio(audio_path, model, scaler_object, feature_column_names):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Custom audio file not found: {audio_path}")

    custom_features = extract_audio_features(audio_path)
    custom_features_df = pd.DataFrame([custom_features])
    custom_features_df = custom_features_df.reindex(columns=feature_column_names)
    custom_features_scaled = scaler_object.transform(custom_features_df)

    probabilities = model.predict_proba(custom_features_scaled)[0]
    class_names = model.classes_
    best_index = int(np.argmax(probabilities))
    predicted_genre = class_names[best_index]
    confidence = float(probabilities[best_index])

    ranking_df = pd.DataFrame(
        {
            "genre": class_names,
            "confidence": probabilities,
        }
    ).sort_values("confidence", ascending=False)

    return predicted_genre, confidence, ranking_df

CUSTOM_AUDIO_PATH = None

if CUSTOM_AUDIO_PATH is not None:
    predicted_genre, confidence, ranking_df = predict_custom_audio(
        CUSTOM_AUDIO_PATH,
        svm_model,
        scaler,
        feature_columns,
    )

    print(f"Predicted genre: {predicted_genre}")
    print(f"Confidence: {confidence:.2%}")
    display(ranking_df)
else:
    print("Set CUSTOM_AUDIO_PATH to a .wav or .mp3 file path, then run this cell again.")
