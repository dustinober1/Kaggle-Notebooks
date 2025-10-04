# 📊 Kaggle Data Science Notebooks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-green?style=for-the-badge&logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

*A collection of comprehensive data science projects featuring machine learning, NLP, and exploratory data analysis*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Projects](#-projects)
  - [Titanic Survival Prediction](#1-titanic-survival-prediction)
  - [Human vs AI Text Classification](#2-human-vs-ai-text-classification)
  - [Los Angeles Crime Data Analysis](#3-los-angeles-crime-data-analysis)
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

### 1. 🚢 Titanic Survival Prediction

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

#### 📊 Results
- **Target Accuracy:** ≥ 98%
- Extensive feature engineering pipeline
- Detailed model comparison and evaluation
- Production-ready predictions

#### 🛠️ Technologies
`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `NLP`

---

### 2. 🤖 Human vs AI Text Classification

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

#### 📊 Results
- Detailed comparison between model variants
- Multi-metric assessment (accuracy, precision, recall, F1)
- Statistical significance analysis
- Production deployment recommendations

#### 🛠️ Technologies
`Python` `PyTorch` `Transformers` `DistilBERT` `Pandas` `NumPy` `Matplotlib` `Seaborn`

---

### 3. 🚔 Los Angeles Crime Data Analysis

**Comprehensive Exploratory Data Analysis**

[![Notebook](https://img.shields.io/badge/View-Notebook-blue?style=flat-square)](./Los%20Angeles%20Crime%20Data%20-%20Exploratory%20Data%20Analysis.ipynb)
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

#### 📊 Dataset Information
- **Source:** Los Angeles Police Department
- **Period:** 2020 - Present
- **Records:** 800,000+
- **Features:** 28 columns
- **Update Frequency:** Weekly

#### 🛠️ Technologies
`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Plotly` `Geopandas` `Folium`

---

## 🛠️ Technologies Used

### Core Libraries
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`, `folium`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
- **Deep Learning:** `pytorch`, `transformers`
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
├── Titanic.ipynb                                          # Titanic survival prediction
├── Human vs AI Generated Essays.ipynb                     # AI text detection with DistilBERT
├── Los Angeles Crime Data - Exploratory Data Analysis.ipynb  # LA crime EDA
├── README.md                                              # This file
├── requirements.txt                                       # Python dependencies
└── data/                                                  # Data directory (not tracked)
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

## 📈 Results & Performance

### Titanic Survival Prediction
- **Target Accuracy:** 98%+
- **Features Created:** 30+
- **Models Trained:** 5+
- **Ensemble Methods:** Weighted voting

### Human vs AI Classification
- **Model Architecture:** DistilBERT (cased & uncased)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score
- **Comparative Analysis:** Statistical significance testing
- **Performance:** High classification accuracy

### LA Crime Analysis
- **Data Points:** 800,000+
- **Time Period:** 2020 - Present
- **Visualizations:** 20+ charts and maps
- **Insights:** Actionable patterns for public safety

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

## 📧 Contact

**Dustin Ober**

[![GitHub](https://img.shields.io/badge/GitHub-dustinober1-black?style=flat-square&logo=github)](https://github.com/dustinober1)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?style=flat-square&logo=kaggle)](https://www.kaggle.com/dustinober1)

---

## 🙏 Acknowledgments

- **Kaggle Community** - For datasets and inspiration
- **Los Angeles Police Department** - For crime data
- **Titanic Dataset** - Classic ML benchmark
- **Open Source Contributors** - For amazing libraries and tools

---

<div align="center">

### ⭐ If you find these projects helpful, please consider giving this repository a star!

**Made with ❤️ and ☕ by Dustin Ober**

</div>
