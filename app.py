import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os

st.title('Faculty Achievements: Year-wise Review Analysis')

# File uploader for CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    review_ip = pd.read_csv(uploaded_file)

    # Convert string to datetime
    review_ip['date'] = pd.to_datetime(review_ip['date'], dayfirst=True)  # Assuming day-month-year format

    # Extract year and month
    review_ip['year'] = review_ip['date'].dt.year
    review_ip['month'] = review_ip['date'].dt.month

    # Group by 'year' and calculate count of reviews
    yearly_count = review_ip.groupby('year')['reviewTitle'].count().reset_index()
    yearly_count.columns = ['Year', 'Review Count']

    # Absolute path to the trained model file
    joblib_file = "random_forest_model.pkl"

    # Initialize the model variable
    model = None

    # Check if the model file exists before loading
    if os.path.exists(joblib_file):
        model = joblib.load(joblib_file)
        st.write("Model loaded successfully.")
    else:
        st.write(f"File {joblib_file} not found. Please check the path.")

    # Line chart for yearly review count
    st.subheader('Year-wise Count of Reviews (Line Chart)')
    fig, ax = plt.subplots()
    ax.plot(yearly_count['Year'], yearly_count['Review Count'], marker='^')
    ax.set_title('Year-wise Count of Reviews')
    ax.set_xlabel('Year')
    ax.set_ylabel('Review Count')
    ax.grid(True)
    st.pyplot(fig)

    # Bar chart for yearly review count
    st.subheader('Year-wise Count of Reviews (Bar Chart)')
    fig, ax = plt.subplots()
    ax.bar(yearly_count['Year'], yearly_count['Review Count'])
    ax.set_title('Year-wise Count of Reviews')
    ax.set_xlabel('Year')
    ax.set_ylabel('Review Count')
    ax.grid(True)
    st.pyplot(fig)

    # Input for user to predict the review count for a specific year
    st.subheader('Predict Review Count for a Specific Year')
    if model:
        user_input_year = st.number_input('Enter a year to predict the review count:', min_value=int(yearly_count['Year'].min()), max_value=int(yearly_count['Year'].max() + 10), value=int(yearly_count['Year'].max() + 1))
        user_input = np.array([[user_input_year]])
        predicted_review_count = model.predict(user_input)
        st.write(f"Predicted review count for the year {user_input_year}: {predicted_review_count[0]}")
    else:
        st.write("Model is not loaded, prediction cannot be made.")

    # Top 10 Product Variants by Review Count
    product_counts = review_ip.groupby('productAsin')['reviewTitle'].count().reset_index()
    product_counts.columns = ['Product Variant', 'Review Count']
    product_counts = product_counts.sort_values(by='Review Count', ascending=False)
    top_products = product_counts.head(10)

    st.subheader('Top 10 Product Variants by Review Count')
    fig, ax = plt.subplots()
    ax.bar(top_products['Product Variant'], top_products['Review Count'])
    ax.set_title('Top 10 Product Variants by Review Count')
    ax.set_xlabel('Product Variant')
    ax.set_ylabel('Review Count')
    ax.set_xticklabels(top_products['Product Variant'], rotation=90)
    ax.grid(True)
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to proceed.")
