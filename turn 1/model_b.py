import pandas as pd
import numpy as np

# Dummy data for climate change features and impacts
data = {
    'Temperature': np.random.randint(10, 30, size=100),
    'Rainfall': np.random.randint(500, 1500, size=100),
    'CO2_emissions': np.random.randint(1000, 2000, size=100),
    'Crop_yield': np.random.randint(50, 150, size=100),
    'Drought_risk': np.random.randint(0, 2, size=100)
}
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define the features and the target variable
features = ['Temperature', 'Rainfall', 'CO2_emissions']
target = 'Crop_yield'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

import schedule
import time


def train_and_predict():
    # Train the model
    regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = regressor.predict(X_test)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")

    # Additional logic to store insights or perform other tasks can be added here


# Schedule the model training and prediction to run daily at 8 AM
schedule.every().day.at("08:00").do(train_and_predict)

# Run the scheduler indefinitely
while True:
    schedule.run_pending()
    time.sleep(1)
