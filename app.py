import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------------------------------
# 1. SETUP & LOADING
# --------------------------------------------------------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

# Load the trained model
# Using @st.cache_resource helps load the model only once, making the app faster
@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

model = load_model()

# Load the dataset just to get the unique options for dropdowns
# (We don't need the whole data for prediction, just the list of brands/fuel types)
@st.cache_data
def get_unique_values():
    df = pd.read_csv("data/car_data.csv") # Make sure this path matches your folder
    df["brand"] = df["name"].str.split().str[0] # Create the brand column again
    
    # Get unique lists
    brand_list = df['brand'].unique().tolist()
    brand_list.sort()
    return brand_list

# Try to load brands, or use a default list if file not found
try:
    brand_options = get_unique_values()
except FileNotFoundError:
    st.error("CSV file not found. Using default brand list.")
    brand_options = ['Maruti', 'Hyundai', 'Datsun', 'Honda', 'Tata', 'Chevrolet', 
                     'Toyota', 'Jaguar', 'Mercedes-Benz', 'Audi', 'Skoda', 'Jeep', 
                     'BMW', 'Mahindra', 'Ford', 'Nissan', 'Renault', 'Fiat', 
                     'Volkswagen', 'Volvo', 'Mitsubishi', 'Land', 'Daewoo', 
                     'MG', 'Force', 'Isuzu', 'OpelCorsa', 'Ambassador', 'Kia']

# Hardcoded options for other columns (since they are small and fixed)
fuel_options = ['Diesel', 'Petrol', 'LPG', 'CNG']
seller_options = ['Individual', 'Dealer', 'Trustmark Dealer']
transmission_options = ['Manual', 'Automatic']
owner_options = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']

# --------------------------------------------------------------------------
# 2. UI LAYOUT
# --------------------------------------------------------------------------
st.title("Car Price Predictor")
st.markdown("Enter the details of the car below to get an estimated selling price.")
st.markdown("---")

# Split the layout into two columns for better visibility
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Car Brand", options=brand_options)
    year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015, step=1)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1000)
    fuel = st.selectbox("Fuel Type", options=fuel_options)

with col2:
    seller_type = st.selectbox("Seller Type", options=seller_options)
    transmission = st.radio("Transmission Type", options=transmission_options)
    owner = st.selectbox("Previous Owners", options=owner_options)

st.markdown("---")

# --------------------------------------------------------------------------
# 3. PREDICTION LOGIC
# --------------------------------------------------------------------------
if st.button("Predict Price", type="primary"):
    
    # 1. Structure user input exactly like the training data
    input_data = pd.DataFrame([[brand, year, km_driven, fuel, seller_type, transmission, owner]],
                              columns=['brand', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'])
    
    # 2. Predict (This gives the Log value)
    prediction_log = model.predict(input_data)
    
    # 3. Reverse the Log (Exp) to get actual currency
    prediction_price = np.exp(prediction_log)[0]
    
    # 4. Display Result
    st.success(f"Estimated Price: ₹ {prediction_price:,.2f}")
    
    # Optional: Show a little info box
    st.info(f"You are selling a {year} {brand} with {km_driven:,} km mileage.")