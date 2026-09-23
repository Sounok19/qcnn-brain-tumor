# Hybrid Quantum-Classical Neural Networks for Brain Tumor MRI Classification[cite: 1]

This repository contains an experimental suite exploring the intersection of Quantum Machine Learning (QML) and classical Deep Learning[cite: 1]. It implements Hybrid Quantum Convolutional Neural Networks (QCNNs) to classify Brain Tumor MRI scans[cite: 1]. The supported classes are Glioma, Meningioma, Pituitary, and No Tumor[cite: 1]. A specialized focus of this project is evaluating quantum advantage in noise robustness[cite: 1].

## 🧠 Project Overview

Classical deep learning models often degrade when exposed to noisy training data[cite: 1]. This project investigates whether appending parameterized quantum circuits to classical feature extractors creates a more robust model[cite: 1]. 

The codebase tests these architectures against classical backbones, including MobileNetV2, ResNet50V2, DenseNet121, Xception, and InceptionV3[cite: 1]. It spans from a custom-built TensorFlow Quantum Simulator to 8-qubit integration using PennyLane[cite: 1].

## ✨ Key Features

* **Custom Vectorized TF Quantum Simulator:** Includes a batch-optimized TensorFlow quantum layer (`TFQuantumSimulationLayer`) that bypasses bottlenecks using `tf.einsum`[cite: 1].
* **PennyLane Integration:** Bridges PennyLane `float64` and TensorFlow `float32` gradients using custom Keras mapping layers[cite: 1].
* **Diverse Hybrid Architectures:**
  * **Parallel Hybrid QCNN:** Branches MobileNetV2 alongside a custom 4-qubit quantum simulator[cite: 1].
  * **Advanced QuanvNN:** Funnels a custom CNN base into an 8-qubit filter[cite: 1].
  * **Dense VQC:** Routes ResNet50V2 or DenseNet121 feature extractors through 8-qubit quantum layers[cite: 1].
  * **Dressed QNN:** Pairs a convolutional base with a ring-entangled quantum circuit[cite: 1].
* **Dynamic Noise Stress Testing:** Injects varying levels of Gaussian noise (0% to 15%) dynamically into the training pipeline to test model degradation[cite: 1].
* **Anti-Overtraining Safeguards:** Implements custom callbacks for safe model checkpointing, automated learning rate decay, and early stopping[cite: 1].

## 📂 Dataset

The project utilizes a Brain Tumor MRI Dataset consisting of 4 classes[cite: 1]:
* `glioma`[cite: 1]
* `meningioma`[cite: 1]
* `notumor` (or `no_tumor`)[cite: 1]
* `pituitary`[cite: 1]

The smart dataset finder automatically traverses directories to locate and label images appropriately, handling common naming variations like "no_tumor"[cite: 1]. The default path is configured for Kaggle environments (`/kaggle/input`)[cite: 1].

## 🔬 Experiment Suite

The repository is structured into distinct experimental phases[cite: 1]:

1. **Classical vs. Hybrid Baseline Training:** Trains a combined MobileNetV2 and 4-Qubit simulator model using a frozen backbone followed by a fine-tuned backbone[cite: 1].
2. **Noise Robustness Experiment:** Iteratively trains a pure classical MobileNetV2 against a Hybrid QCNN across 5 noise levels[cite: 1].
3. **Heavyweight Classical Stress Test:** Evaluates Xception, ResNet50V2, and InceptionV3 under the same noise conditions[cite: 1].
4. **PennyLane 8-Qubit Stress Test:** Tests 8-qubit hybrid models (like DenseNet121 + VQC) using pre-extracted feature vectors to reduce computation time during noise runs[cite: 1].

## 📊 Results

The code automatically generates comparative metrics, including confusion matrices and charts plotting classical versus quantum hybrid accuracy degradation[cite: 1]. 

**Example Output from 8-Qubit DenseNet VQC[cite: 1]:**
* **0.0% Noise:** 94.58% Clean Test Accuracy[cite: 1]
* **0.5% Noise:** 94.72% Clean Test Accuracy[cite: 1]
* **5.0% Noise:** 73.33% Clean Test Accuracy[cite: 1]
* **10.0% Noise:** 69.86% Clean Test Accuracy[cite: 1]
* **15.0% Noise:** 59.17% Clean Test Accuracy[cite: 1]
