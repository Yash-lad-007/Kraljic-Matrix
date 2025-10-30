import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- 1. Data Simulation (To make the app runnable without the original CSV) ---
# NOTE: In a real-world scenario, you would upload the preprocessed data or the model object.
# This synthetic data mimics the structure and scales seen in the notebook output.

@st.cache_resource
def load_and_train_model():
    """Loads a synthetic dataset, preprocesses it, and trains the GaussianNB model."""
    # Data is simulated based on the columns used for training in the notebook
    # Features: Lead_Time_Days, Order_Volume_Units, Cost_per_Unit, Supply_Risk_Score, 
    #           Profit_Impact_Score, Environmental_Impact, Single_Source_Risk
    # Target: Kraljic_Category (Strategic, Leverage, Bottleneck, Non-Critical)

    data = {
        'Lead_Time_Days': np.random.randint(7, 90, 1000),
        'Order_Volume_Units': np.random.randint(50, 20000, 1000),
        'Cost_per_Unit': np.random.uniform(10.0, 500.0, 1000),
        'Supply_Risk_Score': np.random.randint(1, 6, 1000),
        'Profit_Impact_Score': np.random.randint(1, 6, 1000),
        'Environmental_Impact': np.random.randint(1, 6, 1000),
        'Single_Source_Risk': np.random.choice([0, 1], 1000),
        'Kraljic_Category': np.random.choice(['Strategic', 'Leverage', 'Bottleneck', 'Non-Critical'], 1000)
    }
    df = pd.DataFrame(data)
    
    # Simulate the critical feature engineering logic from your notebook:
    # 1. Map 'Single_Source_Risk' (already done in simulation, but useful for reference)
    # 2. Define features (X) and target (y)
    X = df[['Lead_Time_Days', 'Order_Volume_Units', 'Cost_per_Unit',
           'Supply_Risk_Score', 'Profit_Impact_Score', 'Environmental_Impact',
           'Single_Source_Risk']]
    y = df['Kraljic_Category']
    
    # Train the GaussianNB model (which had the best performance in your notebook)
    model = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    model.fit(X_train, y_train)

    # Calculate performance for display purposes
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']

    return model, accuracy

# Load and train the model once
model, accuracy = load_and_train_model()

# --- 2. Streamlit UI Configuration ---
st.set_page_config(
    page_title="Kraljic Matrix Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📦 Kraljic Matrix Item Classifier")
st.markdown("Use the controls in the sidebar to define an item's procurement characteristics and instantly predict its Kraljic classification using a pre-trained **Gaussian Naive Bayes** model.")
st.markdown("---")

# --- 3. Sidebar Input Form ---
st.sidebar.header("Input Item Characteristics")

with st.sidebar.form(key='kraljic_form'):
    st.subheader("I. Quantitative Factors")
    lead_time = st.number_input(
        "Lead Time (Days)",
        min_value=1.0, max_value=200.0, value=45.0, step=1.0,
        help="Time needed from order to delivery."
    )
    order_volume = st.number_input(
        "Order Volume (Units)",
        min_value=1.0, max_value=50000.0, value=10000.0, step=100.0,
        help="Total number of units ordered."
    )
    cost_per_unit = st.number_input(
        "Cost per Unit",
        min_value=0.01, max_value=1000.0, value=50.0, step=0.1,
        help="Price of a single unit."
    )

    st.subheader("II. Qualitative Factors (Score 1-5)")
    # Using sliders for the score inputs (1 to 5)
    supply_risk = st.slider(
        "Supply Risk Score",
        min_value=1, max_value=5, value=3, step=1,
        help="How difficult is it to source this item? (1=Low, 5=High)"
    )
    profit_impact = st.slider(
        "Profit Impact Score",
        min_value=1, max_value=5, value=3, step=1,
        help="How much does this item contribute to the final product cost/revenue? (1=Low, 5=High)"
    )
    environmental_impact = st.slider(
        "Environmental Impact Score",
        min_value=1, max_value=5, value=3, step=1,
        help="The item's environmental or sustainability risk. (1=Low, 5=High)"
    )

    st.subheader("III. Sourcing Risk")
    single_source_risk_str = st.selectbox(
        "Single Source Risk",
        options=['No', 'Yes'],
        index=0,
        help="Is this item supplied by only one vendor?"
    )

    # Submission button
    predict_button = st.form_submit_button("Classify Item")

# --- 4. Prediction Logic and Output ---

if predict_button:
    # Convert 'Yes'/'No' to 1/0 as per the notebook preprocessing
    single_source_risk_int = 1 if single_source_risk_str == 'Yes' else 0

    # Create a DataFrame for the prediction
    new_data = pd.DataFrame([[
        lead_time, order_volume, cost_per_unit,
        supply_risk, profit_impact, environmental_impact,
        single_source_risk_int
    ]], columns=[
        'Lead_Time_Days', 'Order_Volume_Units', 'Cost_per_Unit',
        'Supply_Risk_Score', 'Profit_Impact_Score', 'Environmental_Impact',
        'Single_Source_Risk'
    ])

    # Make the prediction
    prediction = model.predict(new_data)[0]

    # Display the result in a prominent box
    st.header("Prediction Result")
    
    # Define color scheme and icons for better visualization
    if prediction == 'Strategic':
        color = 'red'
        icon = '💎'
        strategy = "Form long-term partnerships, ensure supply security, and focus on innovation."
    elif prediction == 'Leverage':
        color = 'green'
        icon = '📈'
        strategy = "Exploit purchasing power, use competitive bidding, and focus on efficiency."
    elif prediction == 'Bottleneck':
        color = 'orange'
        icon = '⚠️'
        strategy = "Focus on reducing vulnerability, secure alternative sources, and manage risk aggressively."
    else: # Non-Critical
        color = 'blue'
        icon = '⚙️'
        strategy = "Streamline procurement, minimize administrative costs, and use catalog purchases."

    st.success(f"## {icon} Predicted Kraljic Category: **{prediction}**")
    st.write(f"The model suggests a **{prediction}** sourcing strategy for this item.")
    st.info(f"**Recommended Sourcing Strategy:** {strategy}")
    st.markdown("---")

# --- 5. Application Information and Model Metrics ---

st.subheader("Model Information")
col1, col2 = st.columns(2)

col1.metric(
    label="Model Used", 
    value="Gaussian Naive Bayes"
)

col2.metric(
    label="Trained Model Accuracy (Synthetic Data)", 
    value=f"{accuracy:.2f}"
)

st.markdown("""
The Kraljic Matrix is a portfolio analysis tool used to segment purchases based on two key dimensions: 
**Profit Impact** (importance of the purchase to the company's financial results) and **Supply Risk** (complexity of the supply market).

| Category | Profit Impact | Supply Risk | Recommended Strategy |
| :--- | :--- | :--- | :--- |
| **Strategic** (💎) | High | High | Partnership and long-term security. |
| **Leverage** (📈) | High | Low | Maximize purchasing power, competitive bidding. |
| **Bottleneck** (⚠️) | Low | High | Ensure supply security, contingency planning. |
| **Non-Critical** (⚙️) | Low | Low | Streamline and automate procurement. |
""")

st.markdown("---")
st.caption("Note: This application uses a synthetic dataset for demonstration purposes. To use your original data, replace the `load_and_train_model` function with your actual data loading and preprocessing steps, or load your saved model object.")
