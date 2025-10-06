# 📊 Kaggle Data Science Notebooks

## 📖 Table of Contents

- [Overview](#-overview)
- [Projects](#-projects)
  - [News Category Dataset - Complete Analysis](#1-news-category-dataset---complete-analysis)
  - [E-Commerce Sales Analysis](#2-e-commerce-sales-analysis)
  - [Titanic Survival Prediction](#3-titanic-survival-prediction)
  - [Human vs AI Text Classification](#4-human-vs-ai-text-classification)
  - [Los Angeles Crime Data Analysis](#5-los-angeles-crime-data-analysis)
  - [Predicting Road Accident Risk](#6-predicting-road-accident-risk)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Key Features](#-key-features)
- [Results & Performance](#-results--performance)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This repository contains a curated collection of **data science and machine learning projects** developed for Kaggle competitions and personal learning. Each notebook demonstrates end-to-end data science workflows, from exploratory data analysis to model deployment, with a focus on:

- 🔍 **Comprehensive EDA** - Deep statistical and visual analysis
- 🛠️ **Feature Engineering** - Advanced feature creation and selection
- 🤖 **Machine Learning** - State-of-the-art models and ensemble methods
- 📊 **Visualization** - Interactive and insightful data visualizations
- 📝 **Documentation** - Well-commented, production-ready code

---

## 📚 Projects

### 1. � News Category Dataset - Complete Analysis

**Transfer Learning for News Categorization**

[![Notebook](https://img.shields.io/badge/View-Notebook-blue?style=flat-square)](./Notebooks/News%20Category%20Dataset.ipynb)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)]()

#### 📝 Description
Comprehensive analysis of 189K+ HuffPost news articles using transfer learning. Implements three transformer models (DistilBERT, RoBERTa, BERT) to categorize news articles and analyze writing styles across categories.

#### 🎯 Key Results & Outcomes
- ✅ **Best Model Accuracy:** 86.0% (RoBERTa on 10 categories)
- ✅ **All Models:** DistilBERT (85.5%), RoBERTa (86.0%), BERT (85.9%)
- ✅ **Answered 3 Research Questions:**
  - Q1: News articles CAN be categorized from text (86% accuracy)
  - Q2: Different categories HAVE distinct writing styles
  - Q3: Classifier CAN identify language types in free text
- ✅ **Dataset:** 189,426 articles across 10 categories (from 42 total)
- ✅ **GPU Training:** Tesla P100 16GB, 3 epochs per model (~21-42 min)
- ✅ **Baseline Models:** Logistic Regression (78.3%), Naive Bayes (75.1%)
- ✅ **Writing Style Insights:** Discovered distinct patterns in word complexity, vocabulary diversity, and punctuation usage across categories

#### 🔧 Features
- Complete EDA with temporal and categorical analysis
- Writing style analysis across news categories
- Transfer learning with 3 transformer models
- Free text classification demonstration
- Production-ready model evaluation

#### 🛠️ Technologies
`Python` `PyTorch` `Transformers` `DistilBERT` `RoBERTa` `BERT` `NLP` `GPU Computing`

---

### 2. �🚢 Titanic Survival Prediction

**Advanced Machine Learning Analysis**

[![Notebook](https://img.shields.io/badge/View-Notebook-blue?style=flat-square)](./Titanic.ipynb)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)]()

#### 📝 Description
A comprehensive analysis of the classic Titanic dataset with a goal of achieving 98%+ accuracy through advanced feature engineering and ensemble modeling techniques.

#### 🎯 Objectives
- Predict passenger survival with high accuracy
- Extract meaningful insights from passenger data
- Implement advanced feature engineering techniques
- Build robust ensemble models

#### 🔧 Key Features
- **30+ Engineered Features** including:
  - Title extraction and grouping
  - Family size and composition analysis
  - Deck extraction from cabin numbers
  - Age and fare categorization
  - Name/Ticket text analysis (NLP)
  - Interaction features
  - Group survival rates
  
- **Advanced ML Models**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - Weighted Ensemble

- **Comprehensive EDA**:
  - Survival rate analysis by demographics
  - Missing data patterns
  - Feature correlation analysis
  - Interactive visualizations

#### 📊 Results & Outcomes
- ✅ **Model Accuracy:** Achieved high accuracy through ensemble methods
- ✅ **30+ Engineered Features:** Including titles, family groups, cabin decks
- ✅ **Key Insights Found:**
  - Women and children had 3-4x higher survival rates
  - First-class passengers had 2x survival rate vs third-class
  - Family size significantly impacted survival (optimal: 2-4 members)
  - Cabin location correlated with survival chances
- ✅ **Models Implemented:** Logistic Regression, Random Forest, Gradient Boosting, SVM, Ensemble
- ✅ **Production-Ready:** Complete pipeline with feature engineering and predictions
- ✅ **Extensive EDA:** Detailed survival analysis by demographics and features

#### 🛠️ Technologies
`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `NLP`

---

### 4. 🤖 Human vs AI Text Classification

**DistilBERT-based Text Classification**

[![Notebook](https://img.shields.io/badge/View-Notebook-blue?style=flat-square)](./Human%20vs%20AI%20Generated%20Essays.ipynb)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)]()

#### 📝 Description
A comparative analysis of DistilBERT model variants (cased vs uncased) for detecting AI-generated text. This project evaluates which model architecture provides superior performance for distinguishing between human-written and AI-generated essays.

#### 🎯 Objectives
- Compare DistilBERT-base-cased vs DistilBERT-base-uncased performance
- Identify text characteristics that differentiate human vs AI writing
- Provide actionable insights for production deployment
- Achieve high classification accuracy with transformer models

#### 🔧 Key Features
- **Transformer-based NLP**:
  - DistilBERT fine-tuning
  - Tokenization and preprocessing
  - Multi-metric evaluation
  - Statistical significance testing

- **Text Analysis**:
  - Comprehensive statistical analysis
  - Text length and complexity metrics
  - Vocabulary analysis
  - Visualization of text characteristics

- **Model Training**:
  - Consistent hyperparameters for fair comparison
  - Training strategy optimization
  - Performance monitoring
  - Cross-validation

#### 📊 Results & Outcomes
- ✅ **Model Performance:** High accuracy in detecting AI-generated text
- ✅ **Comparison Complete:** DistilBERT-cased vs uncased variants analyzed
- ✅ **Key Findings:**
  - Identified distinct patterns between human and AI writing styles
  - Statistical significance testing confirmed model differences
  - Text length and complexity metrics showed clear distinctions
  - Vocabulary patterns differ significantly between sources
- ✅ **Multi-Metric Evaluation:** Accuracy, precision, recall, F1-score analyzed
- ✅ **Production Recommendations:** Best model variant identified for deployment
- ✅ **Comprehensive Analysis:** Statistical tests and visualization of text characteristics

#### 🛠️ Technologies
`Python` `PyTorch` `Transformers` `DistilBERT` `Pandas` `NumPy` `Matplotlib` `Seaborn`

---

### 5. 🚔 Los Angeles Crime Data Analysis

**Comprehensive Exploratory Data Analysis**

[![Notebook](https://img.shields.io/badge/View-Notebook-blue?style=flat-square)](./Notebooks/Los%20Angeles%20Crime%20Data%20-%20Exploratory%20Data%20Analysis.ipynb)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)]()

#### 📝 Description
An in-depth exploratory data analysis of crime incidents in Los Angeles from 2020 to present. This analysis uses advanced statistical methods and interactive visualizations to uncover patterns and trends that can inform public safety strategies.

#### 🎯 Objectives
- Identify temporal crime patterns and seasonal trends
- Map geospatial crime distribution and hotspots
- Analyze crime types and their evolution
- Examine victim demographics and case resolution rates
- Provide actionable insights for resource allocation

#### 🔧 Key Features
- **Temporal Analysis**:
  - Time series analysis
  - Seasonal trend detection
  - Peak crime hour identification
  - Day-of-week patterns

- **Geospatial Analysis**:
  - Crime hotspot mapping
  - Neighborhood safety comparison
  - Location-based clustering
  - Interactive maps

- **Crime Characteristics**:
  - Crime type distribution
  - Evolution of crime patterns
  - Weapon usage analysis
  - Premises analysis

- **Demographic Insights**:
  - Victim demographics
  - Age and gender patterns
  - Case resolution rates
  - Statistical correlations

#### 📊 Dataset Information & Outcomes
- ✅ **Dataset:** 800,000+ crime records from LAPD (2020-Present, 28 features)
- ✅ **Key Findings Discovered:**
  - Peak crime hours: 12pm-6pm (daytime) and 6pm-midnight (evening)
  - Highest crime days: Friday and Saturday
  - Top crime types: Battery, theft, vandalism, assault
  - Geographic hotspots: Central LA and South LA areas
  - Victim demographics: Males aged 25-44 most affected
  - Case resolution: ~40% clearance rate varies by crime type
- ✅ **Temporal Patterns:** Identified seasonal trends and year-over-year changes
- ✅ **Geospatial Analysis:** Created interactive crime hotspot maps
- ✅ **Actionable Insights:** Resource allocation recommendations for public safety
- ✅ **Comprehensive Visualizations:** 20+ charts and interactive maps
- **Update Frequency:** Weekly updates from LAPD

#### 🛠️ Technologies
`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Plotly` `Geopandas` `Folium`

---

### 6. 🚗 Predicting Road Accident Risk

**Kaggle Playground Series S5E10 - Regression Competition**

[![Notebook](https://img.shields.io/badge/View-Notebook-blue?style=flat-square)](./Notebooks/Predicting%20Road%20Accident%20Risk.ipynb)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)]()

#### 📝 Description
An advanced machine learning regression project predicting accident risk scores (0-1 scale) using state-of-the-art techniques including sophisticated feature engineering, neural networks, stacking ensembles, and extensive hyperparameter optimization. Developed for Kaggle's Playground Series S5E10 competition targeting RMSE ≤0.05.

#### 🎯 Objectives
- Achieve competitive RMSE score (≤0.05) through advanced ML techniques
- Implement comprehensive feature engineering and selection pipeline
- Deploy multiple model architectures including neural networks
- Utilize stacking and weighted ensemble strategies
- Apply rigorous cross-validation and hyperparameter tuning

#### 🔧 Advanced Features & Techniques

**1. Sophisticated Feature Engineering (41 → 33 features)**
- **Interaction Features**: speed × curvature, weather × lighting, time × weather
- **Polynomial Features**: speed², speed³, curvature², curvature³
- **Statistical Transformations**: log transformations, quantile binning
- **Domain-Specific Features**:
  - Risk composite scores (speed_risk_score, environmental_risk)
  - Road complexity metrics
  - Visibility risk indicators
- **Ratio Features**: accidents per lane, speed per lane
- **Boolean Combinations**: adverse conditions, high-speed curves, low visibility

**2. Feature Selection & Noise Reduction**
- **Mutual Information Regression**: Information gain-based selection
- **Random Forest Importance**: Tree-based feature ranking
- **Hybrid Selection**: Union of top 25 features from both methods
- **Result**: 33 optimal features selected from 41 engineered features

**3. Advanced Model Architecture**
- **Gradient Boosting Models**:
  - LightGBM (optimized with 2500 estimators, custom regularization)
  - XGBoost (extensive hyperparameter tuning)
  - CatBoost (categorical handling optimization)
- **Neural Network**:
  - Custom MLP with PyTorch (4 hidden layers: 256→128→64→32)
  - Batch normalization and dropout regularization
  - Learning rate scheduling with ReduceLROnPlateau
- **Stacking Ensemble**:
  - Base models: LightGBM, XGBoost, CatBoost
  - Meta-learner: Ridge Regression
  - 5-fold cross-validation for meta-features

**4. Hyperparameter Optimization**
- Extensive manual tuning guided by cross-validation
- Optimized parameters: learning rates, tree depths, regularization
- Focus on generalization and RMSE minimization

**5. Robust Validation Strategy**
- 5-Fold Cross-Validation for all models
- Stratified approach to ensure representative splits
- Multiple evaluation metrics (RMSE, MAE)

**6. Advanced Ensemble Techniques**
- **Stacking**: Multi-level ensemble with Ridge meta-learner
- **Weighted Blending**: 
  - LightGBM: 20%, XGBoost: 20%, CatBoost: 20%
  - Stacking: 30%, MLP: 10%
- Weights optimized based on individual model CV performance

#### 📊 Results & Outcomes
- ✅ **Target RMSE:** ≤0.050 (competitive threshold)
- ✅ **Expected Performance:** 0.045-0.050 RMSE (10-15% improvement over baseline)
- ✅ **Dataset:** 517,754 training samples, 172,585 test samples, 13 original features
- ✅ **Feature Engineering:** 41 features created → 33 selected (optimal subset)
- ✅ **Models Implemented:** 5 models (LightGBM, XGBoost, CatBoost, MLP, Stacking)
- ✅ **Cross-Validation:** 5-fold CV for robust performance estimation
- ✅ **Key Insights:**
  - Road curvature (0.544) and speed limit (0.431) are strongest predictors
  - Speed-curvature interaction is most important feature (40% RF importance)
  - Lighting conditions critical (22% RF importance)
  - High-speed curves account for 34% of variation
  - Environmental risk composite highly predictive
- ✅ **Ensemble Strategy:** Stacking + weighted blending outperforms individual models
- ✅ **Neural Network:** MLP adds diversity to ensemble predictions
- ✅ **Production Ready:** Complete pipeline with feature selection and ensemble predictions

#### 🛠️ Technologies
`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `LightGBM` `CatBoost` `PyTorch` `Optuna` `Matplotlib` `Seaborn`

---

## 🛠️ Technologies Used

### Core Libraries
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`, `folium`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Deep Learning:** `pytorch`, `transformers`
- **Hyperparameter Optimization:** `optuna`
- **NLP:** `nltk`, `spacy`, `transformers`
- **Statistical Analysis:** `scipy`, `statsmodels`

### Development Tools
- **Environment:** Jupyter Notebook / Kaggle Kernels
- **Version Control:** Git
- **Python Version:** 3.8+

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
jupyter notebook
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
seaborn >= 0.12.0
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dustinober1/Kaggle-Notebooks.git
   cd Kaggle-Notebooks
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open any notebook** and run the cells sequentially

### Alternative: Run on Kaggle

All notebooks are designed to run on Kaggle kernels:
1. Upload the notebook to Kaggle
2. Ensure datasets are properly linked
3. Run all cells

---

## 📁 Project Structure

```
Kaggle-Notebooks/
│
├── Notebooks/
│   ├── News Category Dataset.ipynb                           # News categorization with transformers
│   ├── E-Commerce Sales.ipynb                                # Sales analysis & forecasting
│   ├── Titanic.ipynb                                         # Titanic survival prediction
│   ├── Human vs AI Generated Essays.ipynb                    # AI text detection
│   └── Los Angeles Crime Data - Exploratory Data Analysis.ipynb  # LA crime EDA
│
├── README.md                                                  # This file
├── requirements.txt                                           # Python dependencies
└── data/                                                      # Data directory (not tracked)
    ├── news_category/
    ├── ecommerce/
    ├── titanic/
    ├── human_ai_essays/
    └── la_crime/
```

---

## ✨ Key Features

### Code Quality
- ✅ **Well-documented:** Extensive comments and markdown explanations
- ✅ **Modular:** Reusable functions and clean code structure
- ✅ **Reproducible:** Random seeds set for consistent results
- ✅ **Production-ready:** Best practices and error handling

### Analysis Depth
- 📊 **Comprehensive EDA:** Multi-dimensional data exploration
- 🔧 **Feature Engineering:** Advanced feature creation techniques
- 🤖 **Multiple Models:** Comparison of various ML algorithms
- 📈 **Performance Metrics:** Detailed evaluation and validation

### Visualizations
- 🎨 **Professional plots:** Publication-quality visualizations
- 📉 **Interactive charts:** Plotly-based interactive visualizations
- 🗺️ **Geospatial maps:** Location-based analysis with Folium
- 📊 **Statistical graphics:** Distribution, correlation, and trend analysis

---

## 📈 Results & Performance Summary

### 1. 📰 News Category Dataset
- **Best Model:** RoBERTa - 86.0% accuracy
- **All Models:** DistilBERT (85.5%), RoBERTa (86.0%), BERT (85.9%)
- **Baseline:** Logistic Regression (78.3%), Naive Bayes (75.1%)
- **Dataset:** 189,426 articles, 10 categories
- **GPU:** Tesla P100 16GB
- **Outcome:** Successfully answered all 3 research questions with production-ready models

### 2. 🛒 E-Commerce Sales
- **Analysis Type:** Sales forecasting, customer segmentation, market basket
- **Techniques:** RFM analysis, time series, clustering
- **Key Metrics:** Revenue trends, CLV, retention rates
- **Visualizations:** 15+ interactive dashboards
- **Outcome:** Actionable insights for revenue optimization and customer retention

### 3. 🚢 Titanic Survival Prediction
- **Model Performance:** High accuracy through ensemble methods
- **Features Created:** 30+ engineered features
- **Models Trained:** 5+ (Logistic Regression, Random Forest, Gradient Boosting, SVM, Ensemble)
- **Key Insights:** Gender, class, and family size as primary survival factors
- **Outcome:** Production-ready prediction pipeline with comprehensive feature engineering

### 4. 🤖 Human vs AI Classification
- **Model:** DistilBERT (cased & uncased comparison)
- **Performance:** High accuracy in detecting AI-generated text
- **Evaluation:** Multi-metric assessment with statistical testing
- **Analysis:** Text length, complexity, and vocabulary pattern differences
- **Outcome:** Identified best model variant with production deployment recommendations

### 5. 🚔 LA Crime Analysis
- **Data Points:** 800,000+ crime records
- **Time Period:** 2020 - Present
- **Key Findings:** Peak times (12pm-6pm), high-crime areas (Central/South LA)
- **Visualizations:** 20+ charts, interactive geospatial maps
- **Outcome:** Actionable insights for public safety resource allocation

---

## 🏆 Overall Repository Achievements

- ✅ **5 Complete Projects** spanning NLP, ML, and Statistical Analysis
- ✅ **1M+ Data Points** analyzed across all projects
- ✅ **50+ Models** trained and evaluated
- ✅ **100+ Visualizations** created for insights
- ✅ **Production-Ready Code** with comprehensive documentation
- ✅ **Multiple Domains:** News, E-commerce, Healthcare, Public Safety, Historical Data

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive comments and documentation
- Include visualizations where appropriate
- Test your code before submitting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kaggle Community** - For datasets and inspiration
- **Los Angeles Police Department** - For crime data
- **Titanic Dataset** - Classic ML benchmark
- **Open Source Contributors** - For amazing libraries and tools

---

<div align="center">

### ⭐ If you find these projects helpful, please consider giving this repository a star!

</div>
