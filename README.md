# Triplet Loss Model for Celebrity Face Recognition

This project implements a deep learning model using triplet loss for facial recognition on a 5-celebrity dataset.

---

## Dataset

The model is trained and evaluated on the [5-Celebrity Faces Dataset](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset), which includes:
- **Train dataset**: 5 classes of celebrity faces.
- **Validation dataset**: Separate images for testing.

Structure:
```
datasets/
  train/
    ben_afflek/
    elton_john/
    jerry_seinfeld/
    madonna/
    mindy_kaling/
  val/
    ben_afflek/
    elton_john/
    jerry_seinfeld/
    madonna/
    mindy_kaling/
```

---

## Features

- **Triplet Loss**: Trains the network using anchor, positive, and negative samples.
- **Data Augmentation**: Random flips, rotations, and color adjustments.
- **Visualization**: Displays embeddings and triplet relationships.
- **Fine-tuned Backbone**: Utilizes ConvNeXt Large pretrained on ImageNet.

---

## Installation

### Clone Repository
```bash
git clone https://github.com/mahdif355/Face_recognition.git
cd Face_recognition
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Dataset
```python
import kagglehub
path = kagglehub.dataset_download("dansbecker/5-celebrity-faces-dataset")
```

### Extract Dataset
```python
import zipfile, os

with zipfile.ZipFile('./datasets/5-celebrity-faces-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('./datasets/5-celebrity-faces-dataset')
```

---

## Training

Run the training script:
```bash
python train.py
```
Hyperparameters:
- `IMG_SIZE`: 128
- `BATCH_SIZE`: 32
- `EPOCHS`: 50
- `LEARNING_RATE`: 1e-4
- `MARGIN`: 1.0

The model is saved in the `models/` directory.

---

## Evaluation

Visualize triplets and similarity scores:
```python
from visualization import visualize_triplet
visualize_triplet(encoder, anchor_path, positive_path, negative_path, preprocess)
```

---

## Results

Training Progress:
- Test Accuracy: Up to **92%**
- Test Metric: **1.0** similarity.

---

## Folder Structure
```
.
├── datasets/                # Contains train and val datasets
├── models/                  # Saved model weights
├── resources/               # Triplet pickle files
├── train.py                 # Training script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

---

## License
This project is licensed under the MIT License.

