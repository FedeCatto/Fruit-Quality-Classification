# 🍎 Fruit Quality Classification

This project focuses on the **automatic classification of fruit quality** using **Deep Learning** and **Computer Vision** techniques.  
The objective is to classify fruits into three quality categories:

- **Good**
- **Imperfect**
- **Bad (spoiled)**

The project addresses real-world challenges such as **class imbalance**, limited data availability for some fruit types, and **model interpretability**.

---

## 📌 Project Overview

The system analyzes images of fruits belonging to **16 different categories** (including apples, strawberries, lemons, and pomegranates) to assess their quality and conservation status.

A key challenge tackled in this project is **class imbalance**, where the *bad* quality class represents only **4.99%** of the entire dataset.

---

## 📊 Dataset

The final dataset consists of **1,484 images**, aggregated from the **Harvard Dataverse repository**.

### Dataset Characteristics
- **Label Extraction**:  
  Labels are automatically extracted from image filenames through a preprocessing pipeline.
- **Class Distribution**:
  - Good: **51.48%**
  - Imperfect: **43.53%**
  - Bad: **4.99%**
- **Fruit Variety**:  
  Apples are the most represented fruit, while other varieties (e.g. oranges) contain fewer than 20 samples.

---

## 🛠️ Preprocessing & Data Augmentation

To improve robustness and generalization, the following pipeline was implemented:

### 1. Standardization
- Images resized to **224×224**
- Converted to **RGB**
- Normalized using **ImageNet statistics**

### 2. Data Augmentation (Training Set Only)
- Random rotations  
- Horizontal flips  
- Color jittering (brightness, contrast, saturation)  
- Random resized crops  

These transformations simulate real-world variations in lighting and positioning.

### 3. Stratified Data Split
- Training: **70%**
- Validation: **15%**
- Test: **15%**

Class proportions are preserved across all subsets.

---

## 🤖 Models & Architectures

Two main approaches were compared:

### 🔹 Baseline Model – ResNet18
- Used as a **feature extractor**
- Frozen convolutional layers
- **90% overall accuracy**
- Limited performance on the minority *bad* class

### 🔹 Advanced Model – EfficientNet-B0
- Transfer Learning with **two-stage fine-tuning**
- **Weighted Cross-Entropy Loss** to penalize errors on the *bad* class
- Significant improvement in minority class detection

---

## 📈 Results

| Metric | Value |
|------|------|
| Validation Accuracy | **94%** |
| Test Accuracy | **91%** |
| Bad Class Detection | Improved significantly |

The advanced model successfully overcomes the limitations of the baseline, particularly in identifying spoiled fruits.

---

## 🔍 Model Interpretability (Grad-CAM)

To make the model explainable, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was applied.

The visualizations show that the network focuses on:
- Fruit edges
- Surface texture
- Shape irregularities  

while ignoring irrelevant background elements.

**Intuition:**  
Grad-CAM can be imagined as a flashlight: the red and brighter areas indicate where the model is “looking” when deciding whether a fruit is good or spoiled.

---

## 🚀 Future Work

- Expand the dataset for under-represented classes
- Develop **multi-task models** to predict both fruit type and quality
- Optimize lightweight models for **edge deployment** in agriculture or industry

---

