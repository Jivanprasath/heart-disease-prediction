# ❤️ Heart Disease Prediction System
Django-based heart disease prediction system using machine learning

### (Machine Learning + Django Web Application)

## 📌 Overview
This project is a **Heart Disease Prediction System** built using  
**Machine Learning models** and deployed through a **Django web application**.

The system allows users to enter medical parameters through a web interface
and predicts the likelihood of heart disease using pre-trained ML models.

---

## 🧠 Problem Statement
Heart disease is one of the leading causes of death worldwide.
Early detection can significantly improve treatment outcomes.

This project aims to:
- Provide a simple web-based prediction system
- Use trained ML models for accurate prediction
- Demonstrate full ML + Web integration using Django

---

## 🛠️ Tech Stack
- **Python**
- **Django**
- **Machine Learning (Scikit-learn)**
- NumPy, Pandas
- HTML, CSS
- Joblib (model loading)

---

## 📂 Project Structure

heart-disease-prediction/
│── manage.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── heart_disease_project/
│ ├── init.py
│ ├── settings.py
│ ├── urls.py
│ ├── asgi.py
│ └── wsgi.py
│
├── predictor/
│ ├── migrations/
│ ├── init.py
│ ├── admin.py
│ ├── apps.py
│ ├── models.py
│ ├── views.py
│ ├── urls.py
│ └── ml_model.py
│
├── templates/
│ └── predict.html
│
├── static/
│ └── css/
│ └── style.css
│
└── model/
├── model_1.pkl
├── model_2.pkl
└── model_3.pkl



---

## ⚙️ Workflow
1. User enters medical details through web form
2. Data is sent to Django backend
3. Pre-trained ML model (`.pkl`) is loaded
4. Prediction is generated
5. Result is displayed on the web interface

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Jivanprasath/heart-disease-prediction.git
cd heart-disease-prediction

```
### 2️⃣ Create a virtual environment (recommended)

```
venv\Scripts\activate

```
### 3️⃣ Install dependencies

```
pip install -r requirements.txt

```
### 4️⃣ Run database migrations

```
python manage.py migrate

```
### 5️⃣ Start the Django server

```
python manage.py runserver

```
### 6️⃣ Open in browser

```

http://127.0.0.1:8000/

```

Model Information

1. Multiple ML models are used (.pkl files)
2. Models are stored inside the model/ directory
3. Loaded using joblib inside ml_model.py

🎯 Learning Outcomes

1. Built an end-to-end ML prediction system
2. Integrated Machine Learning with Django
3. Worked with .pkl models in production
4. Designed frontend-backend interaction
5. Gained experience in deploying ML via web apps

👨‍💻 Author

Jivan Prasath S
B.Tech – Robotics and Artificial Intelligence
SASTRA Deemed University






