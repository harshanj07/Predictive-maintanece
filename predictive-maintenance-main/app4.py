import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from joblib import dump, load
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Machine Predictive Maintenance",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Model Training Function ---
MODEL_PATH = "model.joblib"

def train_and_save_model():
    """Train a DecisionTreeClassifier and save it to a file."""
    # Generate a sample dataset
    X_train, y_train = make_classification(
        n_samples=100,
        n_features=6,
        random_state=42,
        n_informative=4,
        n_redundant=2,
        class_sep=1.5,
    )

    # Train the model
    rfc = DecisionTreeClassifier(random_state=42)
    rfc.fit(X_train, y_train)

    # Save the model to a file
    dump(rfc, MODEL_PATH)

# Check if the model file exists; if not, train and save the model
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# --- Load Model ---
try:
    rfc = load(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# --- Page Title ---
st.title("🛠️ Machine Predictive Maintenance Classification")

st.markdown("""
Welcome to the **Machine Predictive Maintenance App**!  
This tool predicts the likelihood of machine failure based on operational parameters.  
Provide inputs below and click **'Predict Failure'** to get recommendations.
""")

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    selected_type = st.selectbox('📊 Select Operational Type', ['Low', 'Medium', 'High'])
    type_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    selected_type = type_mapping[selected_type]

    process_temperature = st.number_input('🔥 Process Temperature [K]', min_value=0.0, step=0.1)

with col2:
    air_temperature = st.number_input('🌡️ Air Temperature [K]', min_value=0.0, step=0.1)
    rotational_speed = st.number_input('🔄 Rotational Speed [rpm]', min_value=0.0, step=1.0)

with col1:
    torque = st.number_input('⚙️ Torque [Nm]', min_value=0.0, step=0.1)

with col2:
    tool_wear = st.number_input('⏳ Tool Wear [min]', min_value=0.0, step=1.0)

# --- Define Prediction Labels and Recommendations ---
failure_labels = {
    0: {
        "message": "✅ No Failure - Machine is operating normally.",
        "recommendation": "No immediate action is required. Continue regular maintenance as scheduled."
    },
    1: {
        "message": "⚠️ Potential Failure - Maintenance may be required soon.",
        "recommendation": "Schedule preventive maintenance to avoid unexpected downtime."
    },
    2: {
        "message": "❌ Critical Failure - Immediate maintenance required!",
        "recommendation": "Immediate intervention is required to avoid significant damage."
    }
}

# --- Prediction ---
if st.button('🔍 Predict Failure'):
    # Prepare input data for prediction
    input_data = np.array([[selected_type, air_temperature, process_temperature,
                            rotational_speed, torque, tool_wear]])

    try:
        # Perform prediction
        failure_pred = rfc.predict(input_data)[0]

        # Display results
        prediction = failure_labels.get(failure_pred, {
            "message": "❓ Unknown Failure Type",
            "recommendation": "No specific recommendation available."
        })

        st.success(prediction["message"])
        st.info(f"💡 Recommendation: {prediction['recommendation']}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

# --- Footer ---
st.markdown("""
---
Developed by Harshan J| Powered by Streamlit
""")
