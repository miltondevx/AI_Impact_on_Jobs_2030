# AI_Impact_on_Jobs_2030
# 🤖 AI Impact on Jobs (2030) Analysis

## 📌 Project Overview
This project performs an end-to-end machine learning analysis to predict the impact of Artificial Intelligence on various job roles by the year 2030. The project utilizes a dataset containing job characteristics such as AI exposure, automation probability, required skills, and salary data.

The workflow includes extensive Exploratory Data Analysis (EDA), **unsupervised learning (Clustering)** to group jobs based on similarities, and **supervised learning** to predict the `Risk_Category` (High, Medium, Low) of different job titles.

## 📂 Dataset
The dataset (`AI_Impact_on_Jobs_2030.csv`) consists of **3,000 entries** and **18 features**, including:
* **Job Features:** `Job_Title`, `Average_Salary`, `Years_Experience`, `Education_Level`
* **AI Indicators:** `AI_Exposure_Index`, `Tech_Growth_Factor`, `Automation_Probability_2030`
* **Skill Scores:** `Skill_1` through `Skill_10` (e.g., Cognitive vs. Manual skills)
* **Target Variable:** `Risk_Category` (High, Medium, Low)

---

## 📊 Exploratory Data Analysis (EDA)
We cleaned the data and performed statistical analysis to understand the relationships between variables.

### Key Visualizations
**1. Correlation Heatmap**
We analyzed the correlation between numeric features, specifically looking at how `Automation_Probability` correlates with various skills and AI exposure.

![Correlation Heatmap](images/heatmap.png)
*(Replace this text with your actual Heatmap screenshot from the project)*

**2. Distribution of Risk Categories**
A count plot was used to visualize the balance of the target classes (Low, Medium, High Risk).

![Risk Category Distribution](images/risk_distribution.png)
*(Replace this text with your Bar Chart of Risk Categories)*

---

## 🧠 Unsupervised Learning: Clustering
We applied **K-Means Clustering** to find hidden patterns in the job data without using the labels.

### 1. The Elbow Method
We used the Elbow Method to determine the optimal number of clusters (`K`) by minimizing Inertia (Sum of Squared Distances).
* **Optimal K:** 3

![Elbow Method Plot](images/elbow_method.png)

### 2. 3D Cluster Visualization (PCA)
To visualize the 3 clusters, we used **Principal Component Analysis (PCA)** to reduce the data dimensionality to 3 components, explaining **22.32%** of the variance.

![3D PCA Clusters](images/pca_3d_clusters.png)

---

## 🤖 Supervised Learning: Risk Prediction
We trained and evaluated **5 different classification models** to predict the `Risk_Category`.

### Model Performance Summary
We evaluated models based on **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

| Model | Accuracy |
| :--- | :--- |
| **Random Forest** | **100.0%** |
| Naive Bayes | 99.5% |
| SVM | 95.7% |
| Logistic Regression | 94.7% |
| KNN | 75.0% |

![Model Accuracy Comparison](images/model_comparison.png)

### Confusion Matrix (Best Model)
The **Random Forest** Classifier achieved the highest performance. Below is the confusion matrix showing its predictions vs. actual values.

![Random Forest Confusion Matrix](images/rf_confusion_matrix.png)

---

## 🔑 Feature Importance
Using the Random Forest model, we extracted the most important features contributing to AI Job Risk. This helps explain *why* certain jobs are classified as High Risk.

![Feature Importance](images/feature_importance.png)

* **Top Insight:** Factors like `Automation_Probability_2030` and specific skill sets were the strongest predictors of job risk.

---

## 🛠️ Technologies Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (KMeans, PCA, LogisticRegression, RandomForest, SVM, KNN, Naive Bayes)

## 🚀 How to Run
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  Run the Jupyter Notebook or Python script to generate the analysis.

## 🔮 Future Work
* Validate the model on external real-world job data to ensure the 100% accuracy holds up outside this dataset.
* Develop a web interface (using Streamlit) where users can input their job details and get a risk prediction.
