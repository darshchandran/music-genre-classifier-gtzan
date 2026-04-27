# Music Genre Classifier

I make music, so I've always been curious about what actually makes a rock song sound like rock and a jazz track sound like jazz. Is it something measurable? Turns out yes — and you can teach a machine to hear it too.

This project extracts audio features from 30-second clips and classifies them into 10 genres. No deep learning black box — just raw audio math and a well-tuned classifier.

---

## What it does

Takes any audio file (mp3 or wav) as input and predicts its music genre along with a confidence score.

```
Input:  [30-second audio clip]
Output: Genre → Jazz (Confidence: 87.4%)

Input:  [Another audio clip]
Output: Genre → Metal (Confidence: 92.1%)
```

---

## Why I built it

Two reasons — I'm personally into music and wanted a project that connected to something I actually care about. And from an ML perspective, audio classification is a completely different problem from images or text. You're not working with pixels or words — you're working with sound waves, frequencies, and rhythm patterns. Learning Librosa and audio feature extraction opened up a whole new domain for me.

---

## Dataset

**GTZAN Genre Collection** — 1,000 audio clips, 30 seconds each, evenly split across 10 genres.  
Genres: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock  
Available on Kaggle: [GTZAN Dataset — Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

## Tech Stack

- **Python 3.11**
- **Librosa** — audio loading and feature extraction
- **Pandas** — building the feature matrix
- **NumPy** — numerical operations
- **Scikit-learn** — SVM, Random Forest, scaling, evaluation
- **Matplotlib / Seaborn** — waveforms, spectrograms, confusion matrix
- **Jupyter Notebook** — development environment

---

## How it works

### 1. Load & Explore
Loaded audio clips genre by genre. Checked clip lengths and sampling rates. Plotted waveforms for a track from each genre — you can already see visual differences between classical (smooth, dynamic) and metal (dense, chaotic).

### 2. Visualize Audio
For each sample clip:
- **Waveform** — amplitude over time
- **Spectrogram** — frequency content over time (shows which frequencies are active when)
- **MFCC Heatmap** — the feature the model relies on most

### 3. Feature Extraction
This is the core of the project. For every audio file, extracted:

| Feature | What it captures |
|---------|-----------------|
| **MFCCs (40 coefficients)** | Timbral texture — how the sound "feels" |
| **Chroma** | Pitch class distribution — the notes being played |
| **Spectral Centroid** | Where the "center of mass" of frequencies sits — brightness |
| **Zero Crossing Rate** | How often the signal crosses zero — relates to percussiveness |
| **RMS Energy** | Overall loudness and energy of the clip |

Each clip becomes a single row of ~50 numbers. 1000 clips = a 1000×50 feature matrix.

### 4. Build Feature Matrix
Looped through all 1000 audio files, extracted features, stored everything in a Pandas DataFrame with genre labels. This took a few minutes to run — audio processing is slower than text.

### 5. Train-Test Split & Scaling
80/20 split. Applied StandardScaler — features like RMS energy and spectral centroid are on very different scales, so normalizing them is important.

### 6. Train SVM
Support Vector Machine works well here because the feature space is relatively small (50 features) and the classes have reasonably clear boundaries in that space. Used RBF kernel with GridSearchCV to find optimal C and gamma values.

### 7. Train Random Forest
Ran a Random Forest alongside for comparison. Also gives feature importance — turns out MFCCs dominate, which makes sense since they capture the most timbral information.

### 8. Evaluate Both Models

| Model | Accuracy |
|-------|----------|
| SVM (RBF kernel) | 82.5% |
| Random Forest | 78.3% |

SVM wins here. Hardest genres to separate: Rock vs Metal, Country vs Blues — genres that genuinely blend into each other.

### 9. Custom Prediction
Drop in any mp3 or wav file — extracts the same features, runs through the SVM, outputs genre + confidence.

---

## Results

| Metric | SVM | Random Forest |
|--------|-----|---------------|
| Accuracy | 82.5% | 78.3% |
| Best genre (F1) | Classical — 0.96 | Classical — 0.94 |
| Hardest genre | Rock — 0.71 | Rock — 0.68 |

Classical is the easiest to classify by far — it's acoustically very distinct. Rock and Metal are the hardest — even humans disagree on where one ends and the other begins.

---

## How to run it

```bash
# Clone the repo
git clone https://github.com/yourusername/music-genre-classifier.git
cd music-genre-classifier

# Install dependencies
pip install librosa pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch the notebook
jupyter notebook music_genre_classifier.ipynb
```

Download the GTZAN dataset from Kaggle and place it in a `data/genres_original/` folder before running.

---

## Project structure

```
music-genre-classifier/
│
├── music_genre_classifier.ipynb    # Main notebook — run this
├── data/
│   └── genres_original/            # GTZAN dataset (download from Kaggle)
│       ├── blues/
│       ├── classical/
│       └── ...
├── features/
│   └── gtzan_features.csv          # Extracted feature matrix (auto-generated)
├── test_audio/                     # Custom audio files for testing
├── requirements.txt
└── README.md
```

---

## What I learned

- How audio is represented mathematically (waveforms, sampling rate, frequency)
- What MFCCs actually are and why they're the go-to feature for audio ML
- How SVM works with non-linear kernels (RBF) and when it outperforms tree-based models
- Feature extraction as a pipeline — going from raw audio to a structured ML-ready matrix
- Why some classification problems are inherently hard (genre boundaries are fuzzy by nature)

---

## What's next

- Try a CNN on mel-spectrograms instead of handcrafted features — let the model learn features itself
- Add more genres beyond the 10 in GTZAN
- Build a Streamlit app — upload a song clip, get a genre prediction live
- Experiment with longer clip lengths to see if accuracy improves
