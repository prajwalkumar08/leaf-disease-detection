# Coastal & Malnad Leaf Disease Detection 🌿

Automated detection of leaf diseases affecting commercial crops in India's coastal and Malnad regions using deep learning (CNN, InceptionV3, DenseNet121).


## 🧠 Project Overview

This project develops an automated leaf disease detection system for key commercial crops grown in Karnataka’s coastal and Malnad regions. The goal is to help farmers identify crop diseases early through image-based classification, enabling timely and sustainable interventions using three deep learning models:
- Custom Convolutional Neural Network (CNN)
- **InceptionV3**
- **DenseNet121**

---

## 📂 Data Collection & Preparation

- **Sources**: PlantVillage and locally collected leaf images (healthy and diseased) from coconut, arecanut, pepper crops.  
- **Processing**:
  - Resize to uniform dimensions (e.g., 224×224 px)
  - Normalize pixel values
  - Augmentation: rotation, flip, zoom, brightness adjustments
  - Split data into 70% train, 20% validation, 10% test

---

## 🛠️ Models

1. **Custom CNN**  
   - Sequential layers (Conv → Pooling → Dense → Dropout)
   - Final softmax classification layer

2. **InceptionV3 (pre-trained)**  
   - Fine‑tuned using transfer learning  
   - Efficient multi-scale feature extraction

3. **DenseNet121 (pre-trained)**  
   - Dense connectivity for feature reuse  
   - Fine‑tuned layers adapted for leaf disease dataset

---

## 🏃‍♂️ Getting Started

### Prerequisites

- Python 3.x  
- Virtual environment (recommended)
