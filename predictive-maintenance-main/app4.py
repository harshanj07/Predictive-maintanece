import streamlit as st
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # Optional if re-training is needed

# --- Page Configuration ---
st.set_page_config(
    page_title="Machine Predictive Maintenance",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Load Model ---
try:
    rfc = joblib.load('model.joblib')  # Ensure the model exists and is correctly saved
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# --- Page Title ---
st.title("🛠️ Machine Predictive Maintenance Classification")

st.markdown("""
Welcome to the **Machine Predictive Maintenance App**! This tool helps predict the likelihood of failure based on key machine parameters.  
Simply provide the required inputs below and click **'Predict Failure'** to receive maintenance recommendations.
""")

# --- Input Section with Two Columns ---
col1, col2 = st.columns(2)

with col1:
    selected_type = st.selectbox('📊 Select Operational Type', ['Low', 'Medium', 'High'])
    type_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    selected_type = type_mapping[selected_type]

with col2:
    air_temperature = st.number_input('🌡️ Air Temperature [K]', min_value=0.0, step=0.1)

with col1:
    process_temperature = st.number_input('🔥 Process Temperature [K]', min_value=0.0, step=0.1)

with col2:
    rotational_speed = st.number_input('🔄 Rotational Speed [rpm]', min_value=0.0, step=1.0)

with col1:
    torque = st.number_input('⚙️ Torque [Nm]', min_value=0.0, step=0.1)

with col2:
    tool_wear = st.number_input('⏳ Tool Wear [min]', min_value=0.0, step=1.0)

# --- Define Prediction Labels with Recommendations ---
failure_labels = {
    0: {
        "message": "✅ No Failure - Machine is operating normally.",
        "recommendation": "No immediate action is required. Regular maintenance should be continued as scheduled."
    },
    1: {
        "message": "⚠️ Potential Failure - Maintenance may be required soon.",
        "recommendation": (
            "The machine is showing signs of wear. Consider scheduling a preventive maintenance check "
            "to avoid unexpected downtime."
        )
    },
    2: {
        "message": "❌ Critical Failure - Immediate maintenance required!",
        "recommendation": (
            "The machine is in a critical state and could fail soon. Immediate intervention is required to "
            "avoid significant damage or downtime."
        )
    }
}

# --- Prediction Button ---
if st.button('🔍 Predict Failure'):
    # Convert inputs into a NumPy array for prediction
    input_data = np.array([[selected_type, air_temperature, process_temperature,
                            rotational_speed, torque, tool_wear]])

    try:
        # Perform prediction
        failure_pred = rfc.predict(input_data)[0]

        # Extract message and recommendation based on prediction
        prediction = failure_labels.get(failure_pred, {
            "message": "❓ Unknown Failure Type",
            "recommendation": "No specific recommendation available."
        })

        # Display the result with a clear statement and detailed recommendation
        st.success(prediction["message"])
        st.info(f"💡 Recommendation: {prediction['recommendation']}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
