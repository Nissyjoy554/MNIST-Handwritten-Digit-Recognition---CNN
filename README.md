# 🧠 MNIST-Digit-Recognition-CNN

### 📝 Description
Handwritten digit recognition project using **Convolutional Neural Networks (CNN)** on the **MNIST dataset**.  
Built with **TensorFlow** and **Keras**, this model achieves around **98–99% accuracy** in classifying handwritten digits (0–9).

---

## 📘 Overview
This project focuses on recognizing handwritten digits (0–9) from the **MNIST dataset** using a **Convolutional Neural Network (CNN)**.  
It demonstrates how deep learning can effectively handle image classification tasks with high accuracy.

---

## 🚀 Technologies Used
- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Deep Learning (CNN)**

---

## 🎯 Objective
To build and train a **CNN model** that can classify handwritten digits from images with high accuracy.

---

## 📂 Dataset
- **MNIST Dataset** — available directly from `tensorflow.keras.datasets`
- Contains **70,000 grayscale images** of handwritten digits (60,000 for training and 10,000 for testing)
- Each image is **28×28 pixels**

---

## ⚙️ Steps Involved
1. **Import Libraries** – TensorFlow, NumPy, Matplotlib  
2. **Load Dataset** – MNIST dataset from Keras  
3. **Preprocess Data** – Normalize and reshape images  
4. **Visualize Data** – Display sample digits  
5. **Build CNN Model** – Using Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers  
6. **Train Model** – Optimize using Adam and monitor accuracy  
7. **Evaluate Model** – Check test accuracy  
8. **Predict New Samples** – Model predicts random digit images

---

## 🧩 Model Architecture
| Layer Type | Parameters |
|-------------|-------------|
| Conv2D | 32 filters, 3x3 kernel, ReLU |
| MaxPooling2D | 2x2 |
| Conv2D | 64 filters, 3x3 kernel, ReLU |
| MaxPooling2D | 2x2 |
| Flatten | — |
| Dense | 128 neurons, ReLU |
| Dropout | 0.5 |
| Dense (Output) | 10 neurons, Softmax |

---

## 📊 Results
- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~98%
- Displays training and validation accuracy/loss graphs
- Predicts random test digits with high confidence

---

## 🔮 Key Learnings
✅ Understanding of CNNs (Convolution, Pooling, Flattening)  
✅ Hands-on with **TensorFlow/Keras**  
✅ Model evaluation and visualization  
✅ Real-time predictions on unseen data  

---

## 📁 How to Run
1. Open in **Google Colab**
2. Copy the notebook code
3. Run all cells
4. Observe accuracy, graphs, and predictions

---
