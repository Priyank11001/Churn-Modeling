import streamlit as st
import pandas as pd
import pickle

# Load the trained model

model = pickle.load(open('sclf.pkl','rb'))

# Function to make predictions
def predict_churn(age, support_calls, payment_delay, total_spend, last_interaction):
    input_data = pd.DataFrame({
        'Age': [age],
        'Support Calls': [support_calls],
        'Payment Delay': [payment_delay],
        'Total Spend': [total_spend],
        'Last Interaction': [last_interaction]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title('Customer Churn Prediction')

st.header('Enter Customer Details:')

age = st.number_input('Age', min_value=0, max_value=120, value=30)
support_calls = st.number_input('Support Calls', min_value=0, max_value=50, value=5)
payment_delay = st.number_input('Payment Delay', min_value=0, max_value=100, value=18)
total_spend = st.number_input('Total Spend', min_value=0.0, value=1000.0, step=10.0)
last_interaction = st.number_input('Last Interaction', min_value=0, max_value=365, value=17)

if st.button('Predict'):
    prediction = predict_churn(age, support_calls, payment_delay, total_spend, last_interaction)
    if prediction == 1:
        st.write('The model predicts that the customer will **churn**.')
    else:
        st.write('The model predicts that the customer will **not churn**.')

st.write('---')

# Display the data for reference (optional)
# st.subheader('Sample Data')
# st.write(final_df.head())
