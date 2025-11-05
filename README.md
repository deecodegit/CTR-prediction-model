# Predictive Modeling for Click-Through Rate Optimization at ConnectSphere Digital  

## Project Overview  
**ConnectSphere Digital**, an online advertising agency, faced a challenge with inefficient ad spend — too many ads were being shown to users who rarely clicked.  
This project develops a **predictive model** to estimate the likelihood that a user will click on an advertisement based on their **demographic and behavioral data**.  

## Business Objective  
To optimize ad targeting and improve **Click-Through Rate (CTR)** by identifying users who are most likely to engage with ads.  
By predicting click probability, the agency can:  
- Focus budget on **high-probability users**  
- Increase **ad engagement**  
- Improve **Return on Ad Spend (ROAS)**  

## Dataset  
The dataset (`advertising.csv`) includes user demographic and behavioral variables such as:  
| **Feature** | **Description** |
|--------------|----------------|
| Daily Time Spent on Site | Average time a user spends daily on the client’s website |
| Age | Age of the user |
| Area Income | Average income of the user’s residential area |
| Daily Internet Usage | Average daily internet usage of the user |
| Ad Topic Line | Subject of the advertisement |
| City, Male, Country | Demographic attributes |
| Timestamp | Time the ad was served |
| Clicked on Ad | Target variable (1 = Clicked, 0 = Not Clicked) |

## Project Workflow  

### 1️. Data Loading & Exploration  
- Imported the dataset using **Pandas** and analyzed variable distributions.  
- Checked correlations, missing values, and feature relationships using **Seaborn** and **Matplotlib**.  

### 2️. Feature Engineering  
- Extracted time-based features (Hour, Weekday).  
- Encoded categorical columns (e.g., Gender).  
- Scaled numerical features using **StandardScaler**.  

### 3️. Model Development  
Built and compared three models:  
- **Logistic Regression** (Baseline model)  
- **Random Forest Classifier**  
- **XGBoost Classifier**  

### 4️. Evaluation Metrics  
Evaluated using:  
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

### 5️. Model Comparison & Lift Chart  
Visualized model performance and lift curve to measure segmentation strength.  

### 6. Explainability with SHAP  
Used **SHAP values** to understand which features drive user clicks —  
For example:  
> “Daily Time Spent on Site” and “Age” were key predictors.

## Results Summary  

| **Model** | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **ROC AUC** |
|------------|--------------|---------------|-------------|---------------|-------------|
| Logistic Regression | ~0.91 | ~0.89 | ~0.93 | ~0.91 | ~0.96 |
| Random Forest | ~0.96 | ~0.95 | ~0.96 | ~0.95 | ~0.98 |
| XGBoost | ~0.97 | ~0.96 | ~0.97 | ~0.96 | ~0.99 |

**XGBoost** performed best overall, offering the most balanced accuracy and interpretability.

## SHAP Insights  
- **Daily Time Spent on Site** → Strongly increases likelihood of click.  
- **Age** → Older users are less likely to click.  
- **Area Income** → Moderate influence on ad engagement.  
- **Daily Internet Usage** → Indicates user’s overall online activity.  
These insights help marketers segment audiences and refine ad delivery timing and targeting.  

## Business Impact  

Integrating this model into ConnectSphere’s ad system allows:  

- **Efficient ad placement** focused on engaged users  
- **Reduced ad wastage**  
- **Increased CTR** and **conversion rates**  
- **Smarter, data-driven marketing decisions**  

### Integration in Ad System  

The CTR prediction model can be integrated into ConnectSphere’s ad-serving pipeline as a **decision layer** between ad selection and delivery.  

**1. Real-Time Ad Selection**  
- User session data (demographics, time, behavior) is passed to the model.  
- The model predicts the probability of a click for each ad.  

**2. Ad Ranking and Delivery**  
- Ads are ranked by predicted CTR values.  
- The top ad is served to the user, improving relevance and engagement.  

**3. Continuous Learning**  
- New click data is collected to retrain and update the model periodically.  

**Conceptual Architecture:**  
```
(User) --> (Feature Collector) --> (CTR Prediction API) --> (Ad Ranking Engine) --> (Ad Display)
```

## Tech Stack  

**Languages:** Python  
**Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP  
**Tools:** Google Colab, GitHub  

## How to Use This Repository  

### 1️. Clone this repository  
```bash
git clone https://github.com/deecodegit/CTR-prediction-model.git
cd CTR-prediction-model
```
### 2. Open the Notebook
Open (`CTR_Prediction_ConnectSphere.ipynb`) in Google Colab or Jupyter Notebook.

### 3. Run all Cells
The notebook is fully annotated.

It will:
- Train the models
- Show evaluation metrics
- Generate SHAP plots automatically

### 4️. Output Files
1. (`logistic_pipeline.joblib`) - saved logistic regression model
2. (`test_scored.csv`) - predictions with probabilities

## Future Enhancements
- Integrate real-time data for continuous ad optimization
- Build a dashboard for CTR trend monitoring
- Deploy model using Flask or Streamlit for live ad scoring
