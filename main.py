import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessors():
    # Then in main.py, load it like this:
    model = xgb.XGBRegressor()
    model.load_model('model.json')
    # Load model
    #with open('model.pkl', 'rb') as f:
    #    model = pickle.load(f)
    
    # Load encoders
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load categorical mappings for reference
    mappings = pd.read_csv('categorical_mappings.csv')
    
    return model, encoders, scaler, mappings

# Set page configuration
st.set_page_config(
    page_title="Singapore HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Main title
st.title("Singapore HDB Resale Price Predictor 🏠")
st.write("""
This application predicts HDB resale flat prices based on various features.
Please fill in the details below to get a prediction.
""")

try:
    # Load model and preprocessors
    model, encoders, scaler, mappings = load_model_and_preprocessors()
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        # Town selection
        town = st.selectbox(
            'Town',
            sorted(encoders['town'].classes_)
        )
        
        # Flat type selection
        flat_type = st.selectbox(
            'Flat Type',
            sorted(encoders['flat_type'].classes_)
        )
        
        # Flat model selection
        flat_model = st.selectbox(
            'Flat Model',
            sorted(encoders['flat_model'].classes_)
        )
        
        # Floor area
        floor_area = st.number_input(
            'Floor Area (square meters)',
            min_value=20.0,
            max_value=200.0,
            value=85.0
        )
    
    with col2:
        # Storey range
        storey_min = st.number_input('Minimum Storey', min_value=1, max_value=50, value=1)
        storey_max = st.number_input('Maximum Storey', min_value=1, max_value=50, value=5)
        
        # Lease commence date
        lease_year = st.number_input(
            'Lease Commence Year',
            min_value=1960,
            max_value=datetime.now().year,
            value=2000
        )
        
        # Calculate derived features
        storey_median = (storey_min + storey_max) / 2
        flat_age = datetime.now().year - lease_year
    
    # Add a predict button
    if st.button('Predict Price'):
        # Encode categorical variables
        town_encoded = encoders['town'].transform([town])[0]
        flat_type_encoded = encoders['flat_type'].transform([flat_type])[0]
        flat_model_encoded = encoders['flat_model'].transform([flat_model])[0]
        
        # Calculate price per sqm (will be scaled)
        price_per_sqm = 0  # placeholder value, will be scaled
        
        # Create feature array
        features = pd.DataFrame({
            'town_encoded': [town_encoded],
            'flat_type_encoded': [flat_type_encoded],
            'flat_model_encoded': [flat_model_encoded],
            'floor_area_sqm': [floor_area],
            'storey_median': [storey_median],
            'flat_age': [flat_age],
            'price_per_sqm': [price_per_sqm]
        })
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Display prediction
        st.success(f'Predicted Resale Price: SGD {prediction:,.2f}')
        
        # Show additional insights
        st.write("### Property Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Floor Area", f"{floor_area:.1f} sqm")
        
        with col2:
            st.metric("Flat Age", f"{flat_age} years")
        
        with col3:
            st.metric("Price per sqm", f"SGD {(prediction/floor_area):,.2f}")

except Exception as e:
    st.error(f"""
    An error occurred while loading the model or making predictions.
    Please make sure all required files are present and valid.
    
    Error: {str(e)}
    """)

# Add footer with information
st.markdown("""
---
### About this Predictor
This application uses machine learning to predict HDB resale flat prices based on historical transaction data. 
The model takes into account various factors including:
- Location (Town)
- Flat Type and Model
- Floor Area
- Storey Range
- Lease Information

**Note**: Predictions are estimates based on historical data and may not reflect actual market values.
""")

# Add sidebar with additional information
st.sidebar.title("About")
st.sidebar.info("""
This app is designed to help potential buyers and sellers estimate HDB resale flat prices in Singapore.
The predictions are based on a machine learning model trained on historical HDB resale transaction data.
""")

st.sidebar.title("Dataset Source")
st.sidebar.markdown("""
Data source: [Data.gov.sg](https://data.gov.sg)
""")