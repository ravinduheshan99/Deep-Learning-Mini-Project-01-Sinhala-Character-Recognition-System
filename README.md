
# Sinhala Character Recognition with CNN
This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten Sinhala characters. Specifically, it focuses on distinguishing between two characters: **"Ra" and "Ya"**.

## Introduction
This project aims to develop a machine learning model capable of recognizing **handwritten Sinhala characters**. Using a **CNN**, the model has been trained to identify two specific characters, **"Ra" and "Ya"**. The project demonstrates the use of data augmentation, model training, and evaluation using **TensorFlow and Keras within a Google Colab notebook.**.

---

<div style="display: flex; justify-content: center; align-items: center;">
   <img src="https://github.com/ravinduheshan99/Deep-Learning-Mini-Project-01-Sinhala-Character-Recognition-System/blob/main/assets/img/CNN.png" alt="Intro" width="1200" height="400">
</div>

---

## Dataset
The dataset comprises images of handwritten Sinhala characters. Each character has its own subdirectory within the Training, Validation, and Testing directories.

- Training: 150 images (75 per character)
- Validation: 150 images (75 per character)
- Testing: 50 images (25 per character)

---

<table>
  <tr>
    <td><img src="https://github.com/ravinduheshan99/Deep-Learning-Mini-Project-01-Sinhala-Character-Recognition-System/blob/main/Training/Ra/11.jpeg" alt="letter ra" width="1200" height="400"></td>
    <td><img src="https://github.com/ravinduheshan99/Deep-Learning-Mini-Project-01-Sinhala-Character-Recognition-System/blob/main/Training/Ya/70.jpeg" alt="letter ya" width="1200" height="400"></td>
  </tr> 
</table>

---

## Model Architecture
The CNN model consists of the following layers:

- **Convolutional Layer:** 16 filters, kernel size (3,3), ReLU activation.
- **MaxPooling Layer:** pool size (2,2).
- **Convolutional Layer:** 32 filters, kernel size (3,3), ReLU activation.
- **MaxPooling Layer:** pool size (2,2).
- **Convolutional Layer:** 64 filters, kernel size (3,3), ReLU activation.
- **MaxPooling Layer:** pool size (2,2).
- **Flatten Layer**
- **Dense Layer:** 512 units, ReLU activation.
- **Dropout Layer:** rate 0.5.
- **Dense Layer:** 1 unit, Sigmoid activation.

## Training
The model is trained using the train_model.py script, which includes:

- Data augmentation using ImageDataGenerator
- Early stopping and learning rate reduction callbacks
- Training for 30 epochs with a batch size of 3

## Results
The model achieves high accuracy on the validation set, with detailed training logs and accuracy metrics recorded. Example results include:

- Validation **Accuracy:** Up to **98.67%**
- Validation Loss: Fluctuations observed, indicating potential overfitting on certain epochs
  
## Key Stages

---

<div style="display: flex; justify-content: center; align-items: center;">
   <img src="https://github.com/ravinduheshan99/Deep-Learning-Mini-Project-01-Sinhala-Character-Recognition-System/blob/main/assets/img/01.png" alt="Img 01" width="1200" height="400">
</div>

---

<div style="display: flex; justify-content: center; align-items: center;">
   <img src="https://github.com/ravinduheshan99/Deep-Learning-Mini-Project-01-Sinhala-Character-Recognition-System/blob/main/assets/img/02.png" alt="Img 02" width="1200" height="400">
</div>

---

<table>
  <tr>
    <td><img src="https://github.com/ravinduheshan99/Deep-Learning-Mini-Project-01-Sinhala-Character-Recognition-System/blob/main/assets/img/03.png" alt="Img 03" width="1200" height="400"></td>
    <td><img src="https://github.com/ravinduheshan99/Deep-Learning-Mini-Project-01-Sinhala-Character-Recognition-System/blob/main/assets/img/04.png" alt="Img 04" width="1200" height="400"></td>
  </tr>
</table>

---

## Future Work
Future improvements could include:

- Expanding the dataset to include more characters and samples.
- Implementing more advanced CNN architectures (e.g., ResNet, Inception).
- Exploring transfer learning with pre-trained models.
- Further tuning of hyperparameters.

## Contact
For any inquiries or feedback, please contact:

Email: ravinduheshan99@gmail.com

## Authors
- [@ravinduheshan99](https://github.com/ravinduheshan99)

