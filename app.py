import streamlit as st
import pandas as pd
import time
import io
import joblib


# ----------------------------
# Load the saved model
# ----------------------------
# Save your Final_Model as a joblib file:
# joblib.dump(Final_Model, "final_model.pkl")
model = joblib.load("final_model.pkl")

# Columns used for prediction
input_columns = [
    'account_length',
    'international_plan',
    'voice_mail_plan',
    'number_vmail_messages',
    'total_night_charge',
    'total_intl_calls',
    'total_intl_charge',
    'customer_service_calls',
    'high_service_calls',
    'total_charge'
]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📞 Telecom Churn Prediction App")
st.write("Predict Customer Churn")

tab1, tab2 = st.tabs(["Prediction", "File Prediction"])

# ----------------------------
# TAB 1 — Prediction
# ----------------------------
with tab1:
    st.header("Enter Customer Details to Predict Churn")

    # Create form for input fields
    with st.form("prediction_form"):
        account_length = st.number_input("Account Length", min_value=1, max_value=300)
        international_plan = st.selectbox("International Plan", ["No", "Yes"])
        voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
        number_vmail_messages = st.number_input("Number Vmail Messages", min_value=0, max_value=100)
        total_night_charge = st.number_input("Total Night Charge", min_value=0.0, format="%.2f")
        total_intl_calls = st.number_input("Total Intl Calls", min_value=0, max_value=100)
        total_intl_charge = st.number_input("Total Intl Charge", min_value=0.0, format="%.2f")
        customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=20)
        high_service_calls = 1 if customer_service_calls > 3 else 0
        total_charge = st.number_input("Total Charge", min_value=0.0, format="%.2f")

        submit = st.form_submit_button("Predict")

    if submit:
        # Encode categorical fields manually (as in training)
        international_plan = 1 if international_plan == "Yes" else 0
        voice_mail_plan = 1 if voice_mail_plan == "Yes" else 0

        # Build input dataframe
        input_data = pd.DataFrame([[
            account_length,
            international_plan,
            voice_mail_plan,
            number_vmail_messages,
            total_night_charge,
            total_intl_calls,
            total_intl_charge,
            customer_service_calls,
            high_service_calls,
            total_charge
        ]], columns=input_columns)

        # Predict
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.subheader("📌 Prediction Result:")
        if prediction == 1:
            st.error(f"Customer is likely to CHURN ⚠ (Probability: {proba:.2f})")
        else:
            st.success(f"Customer will NOT churn 👍 (Probability: {proba:.2f})")

# ----------------------------
# TAB 2 — File PREDICTION
# ----------------------------


# ----------------------------
with tab2:
    st.header("Whole Customers Prediction")
    st.write("Upload a CSV, click Predict, and download the results.")

    @st.cache_resource
    def load_model():
        # Example: model = joblib.load('my_model.pkl')
        # For now, we return a dummy string to simulate a loaded model
        model=joblib.load("final_model.pkl")
        return model

    # --- 2. Prediction Logic ---
    def predict_data(model, input_df):
        """

        Simulate a prediction. 
        Replace this logic with: predictions = model.predict(input_df)
        """
        df=input_df.copy()
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        if 'churn' in df.columns:
            df=df.drop(['churn'],axis=1,errors='ignore',inplace=True)
        df['total_calls'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls']
        df['total_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes'] + df['total_intl_minutes']
        df['avg_call_duration'] = df['total_minutes'] / (df['total_calls'].replace(0,1))
        df['high_service_calls'] = (df['customer_service_calls'] > 3).astype(int)
        df['total_charge'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge']
        df.drop(['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes', 'total_minutes','total_eve_charge','total_day_charge'], axis= 1 , inplace= True)
        df.drop(["total_eve_calls", "total_night_calls", "total_calls", 'total_day_calls', "avg_call_duration", "area_code", "state" ], inplace= True, axis= 1)
        df['international_plan'] = df['international_plan'].map({'Yes': 1, 'No': 0})
        df['voice_mail_plan'] = df['voice_mail_plan'].map({'Yes': 1, 'No': 0})
        # Simulate processing time
        with st.spinner('Predicting...'):
            time.sleep(2) 
            
        # Example logic: Create a new 'Prediction' column based on input
        # (Here we just multiply a numeric column by 2 for demonstration)
        processed_data=df.copy()
        churn_prediction=model.predict(processed_data)
        results_df = input_df.copy()
        results_df['prediction'] = churn_prediction
        results_df['prediction'] = results_df['prediction'].map({
            0: 'Will Not Churn',
            1: 'Will Churn'
        })
            
        return results_df

    
    
    # A. UPLOAD BUTTON
    uploaded_file = st.file_uploader("1. Upload your input CSV", type=["csv"])

    if uploaded_file is not None:
        # Read the file to a dataframe
        input_df = pd.read_csv(uploaded_file)
        input_df= input_df.drop(['Churn'],axis=1,errors='ignore')
        st.write("Preview of Uploaded Data:")
        st.dataframe(input_df.head())

        # Initialize session state for results if it doesn't exist
        if 'result_df' not in st.session_state:
            st.session_state.result_df = None

        # B. PREDICT BUTTON
        # We use a button to trigger the model only when the user is ready
        if st.button("2. Predict Output"):
            model = load_model()
            
            try:
                # Run prediction
                result_df = predict_data(model, input_df)
                
                # Save result to session state so it persists
                st.session_state.result_df = result_df
                st.success("Prediction Complete!")
                
                # Show results
                st.write("Preview of Results:")
                st.dataframe(result_df.head())
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

        # C. DOWNLOAD BUTTON
        # This shows up only if results are available in session state
        if st.session_state.result_df is not None:
            # Convert DataFrame to CSV for download
            csv_buffer = io.BytesIO()
            st.session_state.result_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="3. Download Result File",
                data=csv_data,
                file_name="predicted_results.csv",
                mime="text/csv"
            )