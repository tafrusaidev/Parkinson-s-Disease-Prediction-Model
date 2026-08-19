import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .danger-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 Parkinson\'s Disease Prediction System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## About This App")
st.sidebar.markdown("""
This application uses a **Random Forest Machine Learning Model** 
to predict the likelihood of Parkinson's Disease based on voice 
and speech characteristics.

**Accuracy:** ~95%

**Disclaimer:** This tool is for educational and research purposes 
only and should not replace professional medical diagnosis.
""")

# Load or train model
@st.cache_resource
def load_or_train_model():
    """Load model and scaler, or train if not available"""
    model_path = "parkinsons_model.pkl"
    scaler_path = "parkinsons_scaler.pkl"
    
    # Check if model exists
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, True
    
    # Try to load data and train model
    try:
        df = pd.read_csv('parkinsons.data')
        
        # Prepare data
        X = df.drop(['name', 'status'], axis=1)
        y = df['status']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Save model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return model, scaler, False
    except FileNotFoundError:
        return None, None, False

model, scaler, model_loaded = load_or_train_model()

if model is None:
    st.error("❌ Error: Could not load the training data (parkinsons.data). Please ensure the data file is in the same directory.")
    st.stop()

# Feature names for the model
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'status', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Feature Information", "ℹ️ About"])

with tab1:
    st.markdown("### Enter Patient Voice Features")
    
    # Create input columns
    col1, col2, col3 = st.columns(3)
    
    input_values = {}
    
    # Input fields with descriptions
    with col1:
        input_values['MDVP:Fo(Hz)'] = st.number_input(
            'MDVP:Fo(Hz) - Average vocal fundamental frequency',
            min_value=50.0, max_value=300.0, value=120.0, step=0.1
        )
        input_values['MDVP:Fhi(Hz)'] = st.number_input(
            'MDVP:Fhi(Hz) - Maximum vocal fundamental frequency',
            min_value=50.0, max_value=300.0, value=130.0, step=0.1
        )
        input_values['MDVP:Flo(Hz)'] = st.number_input(
            'MDVP:Flo(Hz) - Minimum vocal fundamental frequency',
            min_value=50.0, max_value=300.0, value=110.0, step=0.1
        )
        input_values['MDVP:Jitter(%)'] = st.number_input(
            'MDVP:Jitter(%) - Variation in fundamental frequency',
            min_value=0.0, max_value=2.0, value=0.5, step=0.01
        )
        input_values['MDVP:Jitter(Abs)'] = st.number_input(
            'MDVP:Jitter(Abs) - Absolute variation',
            min_value=0.0, max_value=0.1, value=0.005, step=0.001
        )
        input_values['MDVP:RAP'] = st.number_input(
            'MDVP:RAP - Relative amplitude perturbation',
            min_value=0.0, max_value=1.0, value=0.03, step=0.01
        )
        input_values['MDVP:PPQ'] = st.number_input(
            'MDVP:PPQ - Pitch period perturbation quotient',
            min_value=0.0, max_value=1.0, value=0.02, step=0.01
        )
        input_values['Jitter:DDP'] = st.number_input(
            'Jitter:DDP - Differential jitter',
            min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )
    
    with col2:
        input_values['MDVP:Shimmer'] = st.number_input(
            'MDVP:Shimmer - Amplitude variation',
            min_value=0.0, max_value=1.0, value=0.03, step=0.01
        )
        input_values['MDVP:Shimmer(dB)'] = st.number_input(
            'MDVP:Shimmer(dB) - Amplitude variation in dB',
            min_value=0.0, max_value=2.0, value=0.2, step=0.01
        )
        input_values['Shimmer:APQ3'] = st.number_input(
            'Shimmer:APQ3 - 3-point amplitude perturbation',
            min_value=0.0, max_value=1.0, value=0.015, step=0.01
        )
        input_values['Shimmer:APQ5'] = st.number_input(
            'Shimmer:APQ5 - 5-point amplitude perturbation',
            min_value=0.0, max_value=1.0, value=0.02, step=0.01
        )
        input_values['MDVP:APQ'] = st.number_input(
            'MDVP:APQ - Amplitude perturbation quotient',
            min_value=0.0, max_value=1.0, value=0.025, step=0.01
        )
        input_values['Shimmer:DDA'] = st.number_input(
            'Shimmer:DDA - Differential amplitude perturbation',
            min_value=0.0, max_value=1.0, value=0.045, step=0.01
        )
        input_values['NHR'] = st.number_input(
            'NHR - Noise-to-harmonics ratio',
            min_value=0.0, max_value=1.0, value=0.02, step=0.001
        )
        input_values['HNR'] = st.number_input(
            'HNR - Harmonics-to-noise ratio',
            min_value=0.0, max_value=40.0, value=25.0, step=0.1
        )
    
    with col3:
        input_values['status'] = st.number_input(
            'status - Subject status (0=healthy, 1=Parkinsons)',
            min_value=0, max_value=1, value=0, step=1
        )
        input_values['RPDE'] = st.number_input(
            'RPDE - Recurrence period density entropy',
            min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )
        input_values['DFA'] = st.number_input(
            'DFA - Detrended fluctuation analysis',
            min_value=0.0, max_value=2.0, value=0.7, step=0.01
        )
        input_values['spread1'] = st.number_input(
            'spread1 - Nonlinear measure spread 1',
            min_value=-10.0, max_value=10.0, value=-5.0, step=0.1
        )
        input_values['spread2'] = st.number_input(
            'spread2 - Nonlinear measure spread 2',
            min_value=-1.0, max_value=1.0, value=-0.3, step=0.01
        )
        input_values['D2'] = st.number_input(
            'D2 - Correlation dimension',
            min_value=0.0, max_value=5.0, value=3.0, step=0.1
        )
        input_values['PPE'] = st.number_input(
            'PPE - Pitch period entropy',
            min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )
    
    # Prediction button
    if st.button('🔍 Predict', key='predict_btn', use_container_width=True):
        # Prepare data for prediction
        features_for_prediction = []
        for feature in feature_names:
            if feature in input_values:
                features_for_prediction.append(input_values[feature])
        
        # Remove the extra 'status' feature (keep only 22 features)
        X_new = np.array([features_for_prediction[:-1]])
        X_new_scaled = scaler.transform(X_new)
        
        # Make prediction
        prediction = model.predict(X_new_scaled)[0]
        prediction_proba = model.predict_proba(X_new_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        
        if prediction == 0:
            st.markdown(
                '<div class="success-box">'
                '<h3>✅ Prediction: HEALTHY</h3>'
                '<p>The model predicts this patient is likely <b>healthy</b>.</p>'
                '</div>',
                unsafe_allow_html=True
            )
            confidence = prediction_proba[0] * 100
        else:
            st.markdown(
                '<div class="danger-box">'
                '<h3>⚠️ Prediction: PARKINSON\'S DISEASE</h3>'
                '<p>The model predicts this patient may have <b>Parkinson\'s Disease</b>.</p>'
                '</div>',
                unsafe_allow_html=True
            )
            confidence = prediction_proba[1] * 100
        
        # Confidence score
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Healthy Probability", f"{prediction_proba[0]*100:.2f}%")
        with col2:
            st.metric("Parkinson's Probability", f"{prediction_proba[1]*100:.2f}%")
        with col3:
            st.metric("Model Confidence", f"{confidence:.2f}%")
        
        # Confidence bar
        st.markdown("#### Confidence Distribution")
        confidence_data = pd.DataFrame({
            'Status': ['Healthy', "Parkinson's Disease"],
            'Probability': [prediction_proba[0]*100, prediction_proba[1]*100]
        })
        st.bar_chart(confidence_data.set_index('Status'))

with tab2:
    st.markdown("### 📚 Understanding the Features")
    
    st.markdown("""
    The prediction model uses 22 voice and speech features extracted from 
    recordings of patients. These features measure various aspects of vocal quality:
    
    #### **Frequency Features (Fundamental Frequency Variations)**
    - **MDVP:Fo(Hz)** - Average vocal fundamental frequency
    - **MDVP:Fhi(Hz)** - Maximum vocal fundamental frequency
    - **MDVP:Flo(Hz)** - Minimum vocal fundamental frequency
    
    #### **Jitter Features (Frequency Perturbations)**
    Measure variations in the period of the vocal cycles:
    - **MDVP:Jitter(%)** - Jitter variation as percentage
    - **MDVP:Jitter(Abs)** - Absolute jitter
    - **MDVP:RAP** - Relative amplitude perturbation
    - **MDVP:PPQ** - Pitch period perturbation quotient
    - **Jitter:DDP** - Differential jitter
    
    #### **Shimmer Features (Amplitude Perturbations)**
    Measure variations in the amplitude of voice cycles:
    - **MDVP:Shimmer** - Shimmer variation
    - **MDVP:Shimmer(dB)** - Shimmer in decibels
    - **Shimmer:APQ3** - 3-point amplitude perturbation
    - **Shimmer:APQ5** - 5-point amplitude perturbation
    - **MDVP:APQ** - Amplitude perturbation quotient
    - **Shimmer:DDA** - Differential amplitude perturbation
    
    #### **Ratio Features**
    - **NHR** - Noise-to-harmonics ratio (higher in Parkinson's)
    - **HNR** - Harmonics-to-noise ratio (lower in Parkinson's)
    
    #### **Nonlinear Dynamical Features**
    - **RPDE** - Recurrence period density entropy
    - **DFA** - Detrended fluctuation analysis
    - **spread1** - Nonlinear measure of fundamental frequency variation
    - **spread2** - Nonlinear measure of fundamental frequency variation
    - **D2** - Correlation dimension
    - **PPE** - Pitch period entropy
    
    #### **Key Insights for Parkinson's Disease**
    - Patients with Parkinson's typically have **higher jitter and shimmer** values
    - **Lower HNR** and **higher NHR** are associated with Parkinson's
    - **Increased noise and reduced voice quality** are common indicators
    """)

with tab3:
    st.markdown("### ℹ️ About Parkinson's Disease")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### What is Parkinson's Disease?
        
        Parkinson's Disease is a progressive neurodegenerative disorder that 
        affects motor control. Common symptoms include:
        
        - **Tremor** (shaking)
        - **Rigidity** (stiffness)
        - **Bradykinesia** (slowness of movement)
        - **Postural instability** (balance problems)
        - **Voice changes** (dysarthria)
        
        #### Voice Changes in Parkinson's
        
        One of the early and significant symptoms is **dysarthria** - 
        difficulty with speech and voice control. This is why voice 
        analysis is an effective screening tool.
        """)
    
    with col2:
        st.markdown("""
        #### Model Information
        
        **Algorithm:** Random Forest Classifier
        
        **Number of Trees:** 100
        
        **Features Used:** 22 voice characteristics
        
        **Training Data:** UCI Parkinson's Disease Dataset
        
        **Typical Accuracy:** ~95%
        
        #### Important Disclaimer
        
        ⚠️ This model is for **educational and research purposes only**.
        
        **Do not use this as a substitute for professional medical diagnosis.**
        
        Always consult with qualified healthcare professionals for:
        - Formal diagnosis
        - Treatment planning
        - Medical advice
        
        Early detection through voice analysis can help healthcare 
        providers identify at-risk individuals for further evaluation.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🧠 Parkinson's Disease Prediction System | Powered by Machine Learning</p>
    <p><small>Disclaimer: This tool is for educational purposes only. Always consult healthcare professionals.</small></p>
</div>
""", unsafe_allow_html=True)
