# Skin Cancer Classification

This project aims to classify skin cancer images into two categories: **Benign** (lành tính) and **Malignant** (ác tính) using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## 🚀 Key Features

*   **Deep Learning Model:** Utilizes a custom CNN architecture to effectively extract features and classify skin images.
*   **Data Augmentation:** Implements data augmentation techniques (rotation, shifting, shearing, zooming, and flipping) using `ImageDataGenerator` and custom functions to artificially expand the training dataset and reduce overfitting.
*   **Imbalanced Data Handling:** Calculates and applies class weights during training to ensure the model doesn't become biased towards the majority class.
*   **Early Stopping:** Employs early stopping to halt training if the model's performance on the validation set stops improving, preventing overfitting and saving computational resources.
*   **Comprehensive Evaluation:** Evaluates the model using various metrics, including Accuracy, Loss, Confusion Matrix, and a detailed Classification Report (Precision, Recall, F1-score).

## 📂 Dataset Overview

The dataset is initially provided as an `archive.zip` file containing images organized into `train` and `test` directories.

**Dataset Distribution:**

*   **Training Set:**
    *   Benign: 1440 images
    *   Malignant: 1197 images
*   **Test Set:**
    *   Benign: 360 images
    *   Malignant: 300 images

Images are preprocessed and resized to **128x128 pixels** before being fed into the neural network.

## 🧠 Model Architecture

The custom Convolutional Neural Network (CNN) consists of the following layers:

1.  **Input Layer:** (128, 128, 3)
2.  **Conv2D:** 32 filters, (3, 3) kernel, ReLU activation
    *   BatchNormalization
    *   MaxPooling2D (2, 2)
3.  **Conv2D:** 64 filters, (3, 3) kernel, ReLU activation
    *   BatchNormalization
    *   MaxPooling2D (2, 2)
4.  **Conv2D:** 128 filters, (3, 3) kernel, ReLU activation *(Extracts complex features)*
    *   BatchNormalization
    *   MaxPooling2D (2, 2)
5.  **Flatten**
6.  **Dense:** 128 units, ReLU activation
    *   BatchNormalization
    *   Dropout (0.4) *(Added to reduce overfitting)*
7.  **Dense (Output Layer):** 1 unit, Sigmoid activation *(Binary classification: Benign vs. Malignant)*

The model is compiled using the **Adam optimizer** with a learning rate of `0.0001` and `binary_crossentropy` as the loss function.

## 🛠️ Requirements

To run this project, you will need the following Python libraries installed:

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `tensorflow` (Keras is included)
*   `scikit-learn`
*   `seaborn`
*   `Pillow` (PIL)
*   `scipy`

You can install them using pip:

```bash
pip install pandas numpy matplotlib tensorflow scikit-learn seaborn Pillow scipy
```

## ⚙️ How to Run

1.  **Extract the Dataset:** The notebook starts by extracting `archive.zip` to a `/dataset` directory.
2.  **Run the Notebook:** Execute the cells in `Skin_Cancer_CNN.ipynb` sequentially.
    *   The notebook handles data loading, augmentation visualization, model training, and evaluation.
3.  **Training:** The model is trained for up to 30 epochs with a batch size of 32, monitored by Early Stopping (patience=10).
4.  **Evaluation:** After training, the notebook will display sample predictions, the test accuracy/loss, a confusion matrix, and graphs of the training and validation accuracy/loss over the epochs.

## 📈 Results Processing

The notebook provides several visualizations to help understand data distribution and model performance:
- Random sample images from the training set.
- Comparisons of original vs. manually augmented images.
- A Confusion Matrix heatmap to visualize True Positives, True Negatives, False Positives, and False Negatives.
- Training history plots showing Accuracy and Loss across epochs for both training and validation phases.