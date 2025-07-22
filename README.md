# 💰 AI Salary Predictor Pro

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://employee-salary-prediction001.streamlit.app/)

**Live Demo:**  
🌐 [https://employee-salary-prediction001.streamlit.app/](https://employee-salary-prediction001.streamlit.app/)

![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange?logo=streamlit)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-blue?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production-brightgreen)

---

## 🧑‍🔬 Model Training & Development

The core machine learning model is developed and trained in the Jupyter/Colab notebook:

> **📓 [`Final_Model+Additional_Features.ipynb`](Final_Model+Additional_Features.ipynb)**  
> - Contains all data preprocessing, feature engineering, model training, evaluation, and export steps.
> - **To understand or retrain the model, open and run this notebook in [Google Colab](https://colab.research.google.com/) or Jupyter.**
> - After training, the notebook exports the model as `salary_predictor_final.pkl` for use in the Streamlit app.

---

## 🖥️ Running the Streamlit App

### 1. **Clone the repository:**
```bash
git clone https://github.com/Arup8/Employee-Salary-Prediction.git
cd employee-salary-prediction
```

### 2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### 3. **Add API keys (optional, for extra features):**
- Create a `.streamlit/secrets.toml` file:
  ```toml
  OPENAI_API_KEY = "your-openai-key"
  RAPIDAPI_KEY = "your-rapidapi-key"
  ```

### 4. **Run the app:**
```bash
streamlit run app.py
```

- The app will open in your browser at `http://localhost:8501`.

---

## 🧑‍💻 Usage

### 🏠 Manual Prediction
- Enter your details and get an instant salary prediction with explanations.

### 📊 Batch Prediction
- Upload a CSV file with employee data to predict salaries in bulk.

### 🔍 What-If Analysis
- See how changing your experience, education, or job title affects your salary.

### 🎯 Skill Gap & AI Career Advisor
- Uncover your skill gaps for any target role—see exactly what you need to level up.
- Let the built-in AI career coach suggest personalized career paths, growth strategies, and the best online courses to accelerate your journey!

### 📈 Dashboard
- Explore salary trends and insights with interactive charts.

### ⚖️ Bias Detection
- Check if the model is fair across gender and other groups.

---

## 📊 Model Performance

- **R² Score:** 0.8360
- **Mean Absolute Error:** $9,827.32
- **Bias Status:** Fair across all demographic groups ✅

---

## 🛠️ Tech Stack

| Frontend   | Backend/ML      | Visualization | AI/LLM         | Utilities      |
|------------|-----------------|--------------|----------------|---------------|
| Streamlit  | XGBoost         | Plotly       | OpenAI GPT-3.5 | Pandas, NumPy  |
|            | Scikit-learn    | Seaborn      |                | Joblib         |
|            | Imbalanced-learn|              |                | Fairlearn      |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

**Arup**  
[GitHub](https://github.com/Arup8)

---

## ⭐ Acknowledgements

- [Streamlit](https://streamlit.io/)
- [XGBoost](https://xgboost.ai/)
- [OpenAI](https://openai.com/)
- [Plotly](https://plotly.com/)
- [Fairlearn](https://fairlearn.org/)

---

> **Ready to explore your salary potential? Launch the app and get started!** 🚀
