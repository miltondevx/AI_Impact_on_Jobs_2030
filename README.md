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
<img width="930" height="790" alt="Correlation heatmap (numeric features" src="https://github.com/user-attachments/assets/4a26d78c-c988-4756-879d-cd52de124b4a" />


**2. Distribution of Risk Categories**
A count plot was used to visualize the balance of the target classes (Low, Medium, High Risk).
<img width="490" height="390" alt="Distribution of Automation Risk Categories" src="https://github.com/user-attachments/assets/3c5ad460-f421-41b5-a010-65f55c14995f" />



---

## 🧠 Unsupervised Learning: Clustering
We applied **K-Means Clustering** to find hidden patterns in the job data without using the labels.

### 1. The Elbow Method
We used the Elbow Method to determine the optimal number of clusters (`K`) by minimizing Inertia (Sum of Squared Distances).
* **Optimal K:** 3
<img width="708" height="466" alt="Elbow Method for Optimal Number of Clusters (K)" src="https://github.com/user-attachments/assets/7a79b465-c262-434f-aa99-21f4fbcc4c79" />



### 2. 3D Cluster Visualization (PCA)
To visualize the 3 clusters, we used **Principal Component Analysis (PCA)** to reduce the data dimensionality to 3 components, explaining **22.32%** of the variance.
<img width="645" height="656" alt="3D PCA Visualization of K-Means Clusters" src="https://github.com/user-attachments/assets/52d4b805-52e3-4e70-827a-fc836a8e8f2f" />



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
<img width="474" height="389" alt="Random_forest" src="https://github.com/user-attachments/assets/1b0dbce0-37e3-43dc-af60-13bb637fc4c3" />



---
### Learning carve for Best Model (Random forest)

<img width="789" height="489" alt="Learning curve" src="https://github.com/user-attachments/assets/b34184ce-03ad-4d0a-b1b3-edc83605b76d" />



---
## 🔑 Feature Importance
Using the Random Forest model, we extracted the most important features contributing to AI Job Risk. This helps explain *why* certain jobs are classified as High Risk.
<img width="790" height="489" alt="Top 10 most important features Random Forest" src="https://github.com/user-attachments/assets/ffadd70c-1c5f-4882-a0bb-87ca9c2903f4" />



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
