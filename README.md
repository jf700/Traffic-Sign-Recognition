# Traffic Sign Recognition: A Comparative Study of CNN Architectures

A comparison of three convolutional neural network approaches to classifying German traffic signs (GTSRB benchmark, 43 classes): a custom CNN trained from scratch, VGG16 with two-phase fine-tuning, and ResNet50 with two-phase fine-tuning.

**Key finding**: The smallest model (294K parameters) outperformed the largest (23.6M parameters) on the held-out test set. ResNet50's parameter count exceeded what the dataset could support without overfitting, producing 486 test errors compared to VGG16's 103 and the custom CNN's 145.

## Project Structure

```
TrafficSignRecognition/
├── data/                          # GTSRB dataset (not in repo — see Dataset Information)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Class distribution, image stats
│   ├── 02_pipeline.ipynb          # tf.data pipeline + augmentation sanity checks
│   ├── 03_custom_cnn.ipynb        # Custom CNN training (Kaggle-ready)
│   └── 04_evaluation.ipynb        # Cross-model comparison and analysis
├── src/
│   ├── preprocessing.py           # Image loading, CLAHE, normalization
│   ├── data_pipeline.py           # tf.data pipelines with augmentation
│   └── models/
│       ├── custom_cnn.py          # From-scratch VGG-style CNN
│       ├── vgg16_model.py         # VGG16 + two-phase fine-tuning wrapper
│       └── resnet_model.py        # ResNet50 + two-phase fine-tuning wrapper
├── report/
│   ├── figures/                   # Convergence curves, confusion matrices, misclassifications
│   ├── tables/                    # Comparison table, classification reports
│   └── report_notes.md            # Structured prep notes for the written report
├── models/saved/                  # Trained model weights (not in repo — see below)
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Training was performed on Kaggle's free GPU notebooks (Tesla T4).

### Installation

Clone the repository:

```bash
git clone https://github.com/jf700/Traffic-Sign-Recognition.git
cd Traffic-Sign-Recognition
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

After cloning, create the local model directory:

```bash
mkdir -p models/saved
```

This directory is excluded from version control because trained model files are large (~50-270 MB each). You'll populate it by running the training notebooks (custom CNN) or downloading model weights from Kaggle after training there (VGG16 and ResNet50).

### Dependencies (also in `requirements.txt`)

- `tensorflow>=2.19` — model training and inference
- `numpy`, `pandas` — data manipulation
- `opencv-python` — image preprocessing, CLAHE
- `scikit-learn` — class weights, stratified split, classification reports
- `matplotlib` — plotting
- `jupyter` — running the notebooks

## Dataset Information

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)**:

- **Training set**: 39,209 RGB images across 43 sign classes
- **Test set**: 12,630 held-out images
- **Source**: Originally published by [Ruhr-Universität Bochum](https://benchmark.ini.rub.de/gtsrb_news.html). Convenient redistribution available on [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

### Download instructions

**For local execution**: download from the Kaggle link above and extract into a `data/` directory at the project root. Expected structure after extraction:

```
data/
├── Train/        # 43 subfolders (one per class), each containing images
├── Test/         # All test images in a flat directory
├── Test.csv      # Test image filenames + class labels
└── Meta.csv      # Class metadata
```

**For Kaggle execution**: attach the GTSRB dataset to your notebook through Kaggle's UI. The notebooks expect the dataset at `/kaggle/input/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/`.

The dataset is not included in this repository due to its size.

## How to Run the Code

The project follows a sequential pipeline. Run notebooks in numerical order:

### 1. Data exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Produces class distribution analysis.

### 2. Data pipeline verification

```bash
jupyter notebook notebooks/02_pipeline.ipynb
```

Builds the `tf.data` pipeline and visualizes augmented vs. clean images.

### 3. Train the custom CNN baseline

```bash
jupyter notebook notebooks/03_custom_cnn.ipynb
```

Runs the from-scratch CNN. Saves trained weights to `models/saved/custom_cnn_best.keras` and training history to `models/saved/custom_cnn_history.json`.

### 4. Train VGG16 and ResNet50

These models require ~90 and ~46 minutes of GPU time respectively. The recommended workflow:

1. Push your code to GitHub (this repo)
2. Create a new Kaggle notebook with GPU + Internet enabled
3. Clone this repo in the Kaggle notebook: `!git clone https://github.com/jf700/Traffic-Sign-Recognition.git /kaggle/working/repo`
4. Use the patterns in `notebooks/03_custom_cnn.ipynb` to build a training notebook, swapping in `build_vgg16_model` or `build_resnet50_model` from `src/models/`
5. After training, download the trained `.keras` and `.json` files

Place downloaded files in `models/saved/` locally.

### 5. Run the cross-model evaluation

```bash
jupyter notebook notebooks/04_evaluation.ipynb
```

Loads all three trained models, computes test accuracy, generates confusion matrices and per-class classification reports, and produces the figures in `report/figures/`.

**Important note**: VGG16 and ResNet50 contain `Lambda` layers for preprocessing that don't deserialize cleanly across TensorFlow versions. The evaluation notebook handles this by rebuilding the architecture from source and loading only the weights, rather than using `tf.keras.models.load_model` directly on those files. 

## Results

### Headline numbers (test set, 12,630 images)

| Model       | Parameters | Test Accuracy | Errors | Training Time |
| ----------- | ---------- | ------------- | ------ | ------------- |
| Custom CNN  | 294K       | 98.85%        | 145    | ~3 min        |
| VGG16       | 14.7M      | **99.18%**    | 103    | ~90 min       |
| ResNet50    | 23.7M      | 96.15%        | 486    | ~46 min       |

VGG16 achieved the highest test accuracy. ResNet50, despite having the most parameters, produced more than 3× the errors of the custom CNN, attributable to overfitting (validation/test accuracy gap of 3.17%, compared to 0.68% for VGG16).

### Key qualitative findings

- All three models' most common errors involve visually similar sign pairs (for example: "60 km/h" vs. "80 km/h", "Slippery road" vs. "Beware of ice/snow"). The errors are plausible at 32×32 resolution.
- Class 27 ("Pedestrians") was the worst-classified class for all three models, suggesting dataset-inherent difficulty rather than architecture failure.
- ResNet50's errors are more diffuse across many class pairs, characteristic of overfit memorization rather than predictable failure modes.

### Where to find the full analysis

- Convergence plots: `report/figures/convergence_curves.png`
- Confusion matrices: `report/figures/confusion_matrices.png`
- Misclassified example images: `report/figures/misclassified_examples.png`
- Per-class precision/recall/F1: `report/tables/classification_report_*.txt`
- Full written report: see `report/` directory

## Reproducing the Results

To exactly reproduce the numbers in this README:

1. Use the same dataset version (Kaggle's `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`)
2. Use TensorFlow 2.19 or 2.21 (slight version differences will not meaningfully change results)
3. Use the random seeds set in `src/data_pipeline.py` (RANDOM_SEED=42)
4. Train on a single-GPU machine (multi-GPU may produce slightly different numerics)

Although even with these controls, exact accuracies will vary by about 0.3% across runs due to non-determinism in GPU operations and data augmentation. The qualitative findings (ResNet50 overfitting, VGG16 winning, custom CNN's competitive performance) are robust across runs.

## Contributors

Josh Fuery, Jean Luc Touma, Lance Nguyen