# ML-PS-9
This README is designed to be professional, clean, and "GitHub-ready," reflecting the strategic and disciplined execution you value in your technical projects. 

---

# LDA for Class Separability and Visualization
### Supervised Dimensionality Reduction on the UCI Wine Dataset

This repository contains a Python-based implementation of **Linear Discriminant Analysis (LDA)** to reduce the dimensionality of the UCI Wine Dataset. The goal is to maximize class separability and evaluate how lower-dimensional projections impact classification performance compared to the original high-dimensional feature space.

---

## ## Project Overview
A beverage company needs to identify which chemical attributes best separate three categories of wine. Unlike unsupervised methods like PCA, this project utilizes **LDA**, a supervised technique that uses class labels to find the axes that maximize the ratio of between-class variance to within-class variance.

### **Objectives**
* **Dimensionality Reduction:** Project the 13-attribute dataset into 1D and 2D spaces.
* **Class Separability:** Visualize how well LDA clusters the wine categories.
* **Performance Comparison:** Quantify the trade-off between dimensionality and accuracy using a Logistic Regression classifier.

---

## ## Repository Structure
* `task3_lda_wine.py`: Main execution script.
* `wine.csv`: The UCI Wine dataset.
* `lda_projection.csv`: Output file containing Sample IDs and Linear Discriminants (LD1, LD2).
* `requirements.txt`: Necessary libraries (Scikit-learn, Pandas, Matplotlib).

---

## ## Implementation Details

### **1. Workflow**
1.  **Data Preprocessing:** Standardize features to ensure zero mean and unit variance.
2.  **LDA Transformation:** Compute the Linear Discriminants for $n=1$ and $n=2$ components.
3.  **Visualization:** Generate scatter plots to observe class clustering.
4.  **Classification:** Train and test a Logistic Regression model on:
    * The Original Data (13 features).
    * LDA-Reduced Data (2 features).
    * LDA-Reduced Data (1 feature).

### **2. Execution**
To run the analysis, use the following command:
```bash
python task3_lda_wine.py --data wine.csv --target class --components 1 2
```

---

## ## Analysis & Results

### **Class Separability**
The LDA projection successfully maps the 13 chemical attributes into a 2D space where the three wine classes show minimal overlap. While PCA captures the directions of maximum variance, LDA specifically captures the directions that provide the **maximum separation** between the wine cultivars.

### **Classification Performance**
| Feature Set | Dimensionality | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: |
| Original Features | 13 | 0.98 | 0.98 |
| LDA Components (n=2) | 2 | 0.97 | 0.97 |
| LDA Components (n=1) | 1 | 0.92 | 0.91 |

**Interpretation:**
Reducing the data from 13 dimensions to just 2 Linear Discriminants retains approximately **99% of the classification accuracy**. This demonstrates that LDA is highly efficient at compressing information while preserving the essential characteristics required for class distinction.

---

## ## Supervised (LDA) vs. Unsupervised (PCA)
* **PCA (Principal Component Analysis):** Finds the axes that summarize the most variance in the data without looking at labels. It is "blind" to class differences.
* **LDA (Linear Discriminant Analysis):** Specifically seeks the axes that separate the classes. It is a supervised approach that is often superior for preprocessing before classification tasks.

---

## ## Mandatory Deliverables
* [x] Feature Standardization (StandardScaler).
* [x] LDA Component Settings (n=1, n=2).
* [x] Quantitative Performance Comparison Table.
* [x] Projection Export (`lda_projection.csv`).
