import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json

# Step 1: Data Preparation
data = {
    'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    'Temperature': [28.5, 28.8, 29.0, 29.2, 29.5, 29.7, 30.0, 30.2, 30.5, 30.8],
    'CO2_Emissions': [350, 355, 360, 365, 370, 375, 380, 385, 390, 395]
}
df = pd.DataFrame(data)


# Step 2: Define the Predictive Model
def train_temperature_model(data):
    X = data[['CO2_Emissions']]
    y = data['Temperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test


# Step 3: Automate Model Evaluation and Representation
def evaluate_model_and_represent_knowledge(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Representing knowledge in a dictionary
    model_results = {
        "model_name": "Linear Regression for Temperature Prediction",
        "metrics": {
            "mean_squared_error": mse
        },
        "coefficients": dict(zip(df.columns[1:], model.coef_.tolist())),
        "intercept": model.intercept_.item()
    }

    return model_results


if __name__ == "__main__":
    model, X_test, y_test = train_temperature_model(df)
    model_knowledge = evaluate_model_and_represent_knowledge(model, X_test, y_test)

    # Step 4: Automatically save the results to a JSON file
    result_file_path = "climate_change_model_results.json"
    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(model_knowledge, f, ensure_ascii=False, indent=4)

    print(f"Model results saved to: {result_file_path}")
