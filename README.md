# Nail Disease Detection using Deep Learning (VGG16)

## Project Overview

This project focuses on detecting and classifying nail diseases using deep learning techniques. A transfer learning approach is implemented using the VGG16 architecture pretrained on ImageNet, enabling efficient feature extraction even with a relatively limited medical dataset.

The model is trained to classify nail images into **17 different disease categories**, providing accurate predictions along with performance visualizations.

---

## Objectives

* Build a robust multi-class image classification model
* Leverage pretrained models (transfer learning) for better accuracy
* Handle class imbalance effectively
* Visualize model performance and predictions
* Create a system suitable for real-world medical assistance

---

## Model Architecture

### Base Model

* **VGG16 (ImageNet pretrained)**
* Top layers removed (`include_top=False`)
* Input size: **48 × 48 × 3**

### Fine-Tuning Strategy

* Initially froze all layers
* Unfroze last **8 layers** for domain-specific learning

### Custom Head

* Global Average Pooling
* Batch Normalization
* Dense Layer (256 units, ReLU)
* Dropout (0.5 + 0.6)
* L2 Regularization
* Output Layer (Softmax – 17 classes)

---

## Training Strategy

### Data Preprocessing

* Image resizing to **48×48**
* Pixel normalization (`rescale=1./255`)

### Data Augmentation

* Rotation
* Zoom
* Width & Height Shift
* Shear Transformation
* Horizontal Flip

### Class Imbalance Handling

* Applied **class weights** using `compute_class_weight`

### Optimization

* Optimizer: **Adam (lr = 1e-4)**
* Loss Function: **Categorical Crossentropy**

### Callbacks Used

* **ReduceLROnPlateau**
* **EarlyStopping**

---

## Dataset Structure

```id="d1qk29"
Nail Disease DataSet/
│
├── Train/
│   ├── alopecia_areata/
│   ├── beau_s_lines/
│   ├── bluish_nail/
│   └── ... (17 classes)
│
└── Test/
    ├── alopecia_areata/
    ├── beau_s_lines/
    └── ...
```

---

## Results & Performance

* Achieved strong classification performance across 17 classes
* Validation accuracy improved steadily during training
* Balanced precision and recall across most classes

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Output Visualizations

```id="p4kz8m"
outputs/images/accuracy_loss_curve.png
outputs/images/sample_predictions.png
outputs/images/confusion_matrix.png
outputs/images/final_prediction_output.png
```

### Explanation

* **Accuracy/Loss Curve** → Shows model learning trend
* **Sample Predictions** → True vs Predicted labels
* **Confusion Matrix** → Class-wise performance
* **Final Output** → Disease + affected nail zone + deficiency insight

---

## How to Run the Project

### Clone Repository
```id="c9w21f"
git clone https://github.com/your-username/nail-disease-detection.git
cd nail-disease-detection
```

### Install Dependencies

```id="a8l2k9"
pip install -r requirements.txt
```

### Mount Google Drive (Colab)

```id="v2j91x"
from google.colab import drive
drive.mount('/content/drive')
```

### Train Model

```id="q7p5nm"
python train.py
```

---

## Requirements

* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn

---

## Model Output

Trained model is saved as:

```id="m8qz1l"
nail_disease_model.h5
```

---

## Future Enhancements

* Upgrade to **EfficientNet / ResNet50**
* Add **nail segmentation (zone detection)**
* Integrate **vitamin deficiency prediction**
* Deploy as a **web or mobile application**
* Improve dataset size and diversity

---

## Author

* Final Year Engineering Student
* Passionate about AI in Healthcare

---

## Conclusion

This project demonstrates the effectiveness of transfer learning in medical image classification. With further improvements, it can evolve into a real-world diagnostic support system.
