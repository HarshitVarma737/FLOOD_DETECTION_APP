# 🌊 Smart Flood Detection System
A web-based ML-powered flood detection system that takes environmental inputs and predicts the possibility of a flood using trained models.

## 🚀 Features
- Real-time prediction using a trained **Random Forest Classifier**
- Beautiful UI with **sea-wave animated background**
- Inputs include temperature, rainfall, humidity, wind speed, etc.
- Displays prediction (`Flood` or `No Flood`) and confidence score
- Input validation to ensure all fields are filled

---

## 🧠 ML Models Used
- Random Forest Classifier
- Logistic Regression (model comparison)
- Scikit-learn-based preprocessing and model training

---

#### 🛠 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/flood-detection-app.git
cd flood-detection-app

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the App
python app.py

### Tech Stack
Frontend: HTML, CSS (custom + animations)

Backend: Flask (Python)

ML Framework: Scikit-learn

Model Storage: joblib

