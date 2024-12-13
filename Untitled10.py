#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess the dataset
review_ip = pd.read_csv(r'C:\\Users\\karri\\Downloads\\iphone.csv')
review_ip['date'] = pd.to_datetime(review_ip['date'], dayfirst=True)
review_ip['year'] = review_ip['date'].dt.year
yearly_count = review_ip.groupby('year')['reviewTitle'].count().reset_index()
yearly_count.columns = ['Year', 'Review Count']

# Prepare data for training
X = yearly_count[['Year']]
y = yearly_count['Review Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib_file = "random_forest_model.pkl"
joblib.dump(model, joblib_file)
print(f"Model saved as {joblib_file}")


# In[ ]:




