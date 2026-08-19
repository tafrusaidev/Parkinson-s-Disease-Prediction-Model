<div align="center">

# 🧠 Parkinson's Disease Prediction Model

### Voice-based Machine Learning system for early Parkinson's Disease screening

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

A **Random Forest** classifier that predicts the likelihood of Parkinson's Disease from vocal biomarkers, wrapped in an interactive **Streamlit** web app.

[Overview](#-overview) • [Demo](#-app-preview) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Dataset](#-dataset) • [Disclaimer](#-disclaimer)

</div>

---

## 📌 Overview

Parkinson's Disease often affects a patient's voice long before other motor symptoms become obvious — changes like tremor, reduced loudness, and breathiness show up as measurable variations in pitch, jitter, and shimmer. This project uses that idea to build a **non-invasive, voice-based screening tool**:

- 🎙️ Takes 22 acoustic features extracted from a sustained vowel recording
- 🌲 Feeds them into a **Random Forest Classifier**
- 📊 Returns a prediction (Healthy / Parkinson's) with a confidence score
- 🖥️ All wrapped in a clean, tabbed **Streamlit** interface

> ⚠️ **This is an educational/research tool — not a diagnostic device.** See the [Disclaimer](#-disclaimer) below.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Live Prediction** | Enter voice features manually and get an instant prediction |
| 📈 **Confidence Scores** | Visual probability breakdown (Healthy vs. Parkinson's) with a bar chart |
| 📚 **Feature Guide** | Built-in explanations of every jitter, shimmer, and nonlinear voice feature |
| ℹ️ **Educational Tab** | Background on Parkinson's Disease and how voice analysis helps detect it |
| ⚡ **Auto-training** | Automatically trains and caches the model on first run if no saved model exists |
| 🎨 **Clean UI** | Custom-styled Streamlit interface with color-coded result cards |

---

## 🖼️ App Preview

```
🧠 Parkinson's Disease Prediction System
┌─────────────────────────────────────────────┐
│  🔮 Prediction   📊 Feature Info   ℹ️ About  │
├─────────────────────────────────────────────┤
│  Enter Patient Voice Features                │
│  [ MDVP:Fo(Hz) ] [ Jitter(%) ] [ Shimmer ]   │
│  ...                                         │
│              [ 🔍 Predict ]                  │
├─────────────────────────────────────────────┤
│  ✅ Prediction: HEALTHY                      │
│  Healthy: 92.4%   Parkinson's: 7.6%          │
└─────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
Parkinson-s-Disease-Prediction-Model/
├── app.py                              # Streamlit web application
├── Parkinsons_Disease_Prediction.ipynb # Model exploration & training notebook
├── parkinsons.data                     # UCI voice dataset
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/tafrusaidev/Parkinson-s-Disease-Prediction-Model.git
cd Parkinson-s-Disease-Prediction-Model
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

**Run the Streamlit app:**

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

On first launch, the app automatically trains a Random Forest model on `parkinsons.data` and caches it (`parkinsons_model.pkl`, `parkinsons_scaler.pkl`) for instant reuse afterward.

**Explore the notebook:**

```bash
jupyter notebook Parkinsons_Disease_Prediction.ipynb
```

Use this to see the full data exploration, preprocessing, and model evaluation process.

---

## 🧬 How It Works

1. **Data Preprocessing** — Voice features are scaled using `StandardScaler` to normalize the range of each measurement.
2. **Model Training** — A `RandomForestClassifier` (100 estimators) is trained on labeled samples (`status`: 0 = healthy, 1 = Parkinson's).
3. **Inference** — New voice measurements are scaled the same way, then passed to the model for a class prediction and probability score.
4. **Result Display** — The app renders the prediction, class probabilities, and a confidence bar chart.

### Voice Feature Categories

| Category | Examples | What It Measures |
|---|---|---|
| **Frequency** | `MDVP:Fo`, `MDVP:Fhi`, `MDVP:Flo` | Vocal fundamental frequency (pitch) |
| **Jitter** | `MDVP:Jitter(%)`, `RAP`, `PPQ` | Cycle-to-cycle frequency variation |
| **Shimmer** | `MDVP:Shimmer`, `APQ3`, `APQ5` | Cycle-to-cycle amplitude variation |
| **Noise Ratios** | `NHR`, `HNR` | Noise-to-harmonics balance (voice quality) |
| **Nonlinear Dynamics** | `RPDE`, `DFA`, `spread1/2`, `D2`, `PPE` | Complex, nonlinear voice signal patterns |

📌 Key clinical insight: Parkinson's patients typically show **higher jitter/shimmer**, **higher NHR**, and **lower HNR** — indicating reduced vocal stability and increased noise.

---

## 📊 Dataset

This project uses the **[UCI Parkinson's Disease Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)**, created by Max Little of the University of Oxford in collaboration with the National Centre for Voice and Speech, Denver, Colorado.

- **195 voice recordings** from 31 individuals (23 with Parkinson's, 8 healthy)
- **22 numeric voice features** per recording
- **Target:** `status` (0 = healthy, 1 = Parkinson's)

---

## 🛠️ Tech Stack

- **Python** — core language
- **pandas / numpy** — data handling
- **scikit-learn** — Random Forest model & preprocessing
- **Streamlit** — interactive web app
- **matplotlib / seaborn** — data visualization (notebook)

---

## 🗺️ Roadmap

- [ ] Add voice recording upload with automatic feature extraction
- [ ] Compare additional models (SVM, XGBoost, Neural Networks)
- [ ] Add model explainability (SHAP / feature importance visualization)
- [ ] Deploy live demo on Streamlit Community Cloud
- [ ] Add unit tests and CI pipeline

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This project is intended **strictly for educational and research purposes**. It is **not a certified medical device** and must **not** be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified healthcare provider for concerns related to Parkinson's Disease or any other medical condition.

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute with attribution.

---

<div align="center">

Made with ❤️ by [tafrusaidev](https://github.com/tafrusaidev)

If this project helped you, consider giving it a ⭐!

</div>
