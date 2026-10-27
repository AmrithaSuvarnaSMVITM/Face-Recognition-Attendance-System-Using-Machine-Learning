# Face Recognition Attendance System using Machine Learning

A multi-person attendance system that recognises faces from a webcam, marks attendance only **after the faculty is verified**, rejects unknown faces, and saves **one CSV file per day**.

The project started as a simple AIcademy program (one photo per person, distance matching). It was rebuilt as a complete **Machine Learning** project: dataset collection, cleaning, leakage-free splitting and augmentation, comparison of five classifiers with grouped cross-validation, hyperparameter tuning, held-out testing and unknown-face rejection.

> **Note on the ML/DL split:** face embeddings (128-d vectors) come from the pre-trained deep CNN inside `dlib`. The classifiers trained, compared and tuned in this project are classical ML models on top of those embeddings.

---

## Features

- Recognises **several people in the same frame** at once
- **Faculty-first rule**: students are marked only after the faculty face is confirmed
- Colour-coded frames: **red** = faculty, **green** = student, **yellow** = student seen before faculty (not marked), **orange** = unknown
- **Unknown-face rejection** using a probability threshold and a nearest-distance check
- Must be recognised in 3 consecutive checks before being marked (avoids false marks)
- Live side panel with subject, faculty status, and each student's status and time
- **One CSV per day** (`attendance_YYYY-MM-DD.csv`); repeated takes on the same day go into the same file
- Runs on a CPU-only laptop (half-size frames, recognition on every 2nd frame)

## Results

Dataset: 5 people (1 faculty, 4 students), 209 usable real photos, 80/20 split of real photos. Training set: 341 samples (168 real + 173 augmented). Test set: 41 real photos (never augmented).

| Model | CV accuracy (5-fold) | CV F1 | Test accuracy | Test F1 | ms / face |
|---|---|---|---|---|---|
| **Logistic Regression** | **97.63% ± 1.51** | **0.9751** | 100% | 1.000 | 0.015 |
| SVM (linear) | 97.32% ± 1.76 | 0.9723 | 97.56% | 0.973 | 0.014 |
| SVM (RBF) | 97.04% ± 1.60 | 0.9695 | 100% | 1.000 | 0.047 |
| Random Forest | 97.04% ± 1.60 | 0.9684 | 100% | 1.000 | 0.152 |
| KNN | 96.74% ± 1.74 | 0.9659 | 100% | 1.000 | 0.019 |

- **Selected model:** Logistic Regression, tuned with `GridSearchCV` (best `C = 10`, CV F1 = 0.9781).
- **Unknown rejection:** 100% of known test faces accepted, 100% of 3 unseen unknown photos rejected.

**How to read these numbers:** all five models are within about 1% of each other, which is smaller than the cross-validation standard deviation, so they perform comparably. Logistic Regression was chosen for the best CV F1, fast prediction, simplicity and calibrated probabilities. The 100% test score is on only 41 photos taken in the same sessions as the training photos, so the cross-validated accuracy (~97.6%) is the more cautious figure.

Charts are generated in `results/`: `model_comparison.png`, `confusion_matrix.png`, plus `model_comparison.csv` and `classification_report.txt`.

## How it works

```
Webcam -> face detection (HOG + SVM) -> landmarks / alignment -> 128-d embedding (dlib)
       -> ML classifier (Logistic Regression) -> probability + nearest-distance check
       -> known?   -> faculty-first rule -> mark attendance -> daily CSV
       -> unknown? -> orange frame, never marked
```

1. **Capture** real photos per person (several sessions, different lighting and angles).
2. **Build dataset**: encode photos, move photos with no detectable face to `dataset/rejected/`, split real photos 80/20, **augment only the training photos** (brightness, contrast, gamma, rotation, flip, blur, noise), save `features.npz`.
3. **Train and compare**: KNN, SVM (linear), SVM (RBF), Logistic Regression, Random Forest in a `StandardScaler` pipeline, evaluated with 5-fold `StratifiedGroupKFold` (augmented copies of one photo stay in the same fold). Tune the best model, evaluate once on the test set, save `model.pkl`.
4. **Run attendance** with the saved model.

### Avoiding data leakage
- Split real photos **before** augmenting
- Test photos are never augmented
- Cross-validation uses group ids so copies of one photo never sit in different folds

## Project structure

```
.
├── capture.py            # collect photos from the webcam
├── build_dataset.py      # clean, split, augment, extract embeddings
├── train_compare.py      # compare models, tune, evaluate, save model.pkl
├── attendance.py         # live attendance system
├── requirements.txt
├── dataset/
│   ├── raw/<name>/       # real photos, one folder per person
│   ├── unknown_test/     # strangers, used only to test Unknown rejection
│   └── rejected/         # photos where no face was found
├── results/              # charts, comparison table, classification report
├── features.npz          # generated
├── model.pkl             # generated
└── attendance_YYYY-MM-DD.csv   # generated, one per day
```

## Installation

Python 3.8+ and a webcam are required.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`face_recognition` depends on `dlib`, which can be hard to install on Windows. If the import fails, install a pre-built dlib wheel, or install CMake and Visual Studio Build Tools and reinstall.

## Usage

```bash
# 1. Capture photos (run once per person; s = save, q = quit). Use lowercase names.
python capture.py

# 2. (Optional) put photos of strangers in dataset/unknown_test/

# 3. Build the dataset
python build_dataset.py

# 4. Train, compare and save the best model
python train_compare.py

# 5. Take attendance (enter the subject, press q to save and quit)
python attendance.py
```

Tips for good data: 40+ photos per person, taken in several sessions (window light, room light, another room), varied head angles and expressions, one face per photo. Re-run steps 3 and 4 whenever you add a person or change photos.

## Attendance CSV

One file per date, e.g. `attendance_2026-10-24.csv`:

```csv
Date,Name,Role,Subject,Status,Time In,Confidence (%)
2026-10-24,Amritha,Faculty,ML,Present,10:00:00,99.0
2026-10-24,Ashish,Student,ML,Present,10:05:00,95.0
2026-10-24,Shaila,Student,ML,Absent,,
```

| Situation on the same day | Result |
|---|---|
| Same subject taken again | Rows are updated (Absent can become Present, never the reverse); no duplicates |
| Different subject | New rows are added |
| File open in Excel when saving | Asks you to close it, then saves into the same file |
| New date | A new file is created |

## Configuration (top of `attendance.py`)

| Setting | Default | Meaning |
|---|---|---|
| `FACULTY_NAMES` | `["amritha"]` | Must match the folder name in `dataset/raw` |
| `PROB_THRESHOLD` | `0.60` | Minimum model confidence |
| `DIST_THRESHOLD` | `0.55` | Max distance to the nearest known face |
| `CONFIRM_FRAMES` | `3` | Consecutive checks before marking |
| `SCALE` | `0.5` | Frame shrink factor |
| `PROCESS_EVERY` | `2` | Recognise on every N-th frame |

## Limitations

- Small dataset (5 people); the test set has only 41 photos and unknown rejection was tested on only 3 photos
- No liveness or anti-spoofing: a printed photo or phone screen may be accepted
- Accuracy depends on lighting, camera quality and distance
- The model must be retrained when a new person is enrolled

## Future work

- More people and photos from different days for a stronger test set
- Liveness / blink detection
- Database storage and a dashboard
- Fixed classroom camera support
- A separate deep-learning version (fine-tuned CNN) for comparison

## Privacy and GitHub

Face images are personal data. Take consent before photographing anyone, and **do not upload** photos or trained models that contain people's face data. Add this to `.gitignore`:

```
dataset/
features.npz
model.pkl
attendance_*.csv
failed_images.txt
.venv/
__pycache__/
```

Add a few screenshots (with consent) or the `results/` charts to the repository instead.

## Tech stack

Python · face_recognition (dlib) · OpenCV · scikit-learn · NumPy · pandas · matplotlib · joblib

## Acknowledgements

Built on the open-source `face_recognition` and `dlib` libraries. Originally developed as an AIcademy project and upgraded into a Machine Learning project.
