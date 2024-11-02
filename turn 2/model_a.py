import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import os
import time

# Function to load data from CSV
def load_data(file_path):
	if os.path.exists(file_path):
		return pd.read_csv(file_path)
	else:
		return pd.DataFrame()

# Function to save data to CSV
def save_data(data, file_path):
	data.to_csv(file_path, index=False)

def train_temperature_model(data):
	X = data[['CO2_Emissions']]
	y = data['Temperature']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	model = LinearRegression()
	model.fit(X_train, y_train)

	return model, X_test, y_test

def evaluate_model_and_represent_knowledge(model, X_test, y_test):
	y_pred = model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	print(f"Mean Squared Error: {mse}")

	model_results = {
		"model_name": "Linear Regression for Temperature Prediction",
		"metrics": {"mean_squared_error": mse},
		"coefficients": dict(zip(df.columns[1:], model.coef_.tolist())),
		"intercept": model.intercept_.item(),
	}

	return model_results

def update_model_periodically(data_file_path, result_file_path, update_interval_sec=86400):
	while True:
		try:
			# Step 1: Load historical data
			data = load_data(data_file_path)

			if not data.empty:
				# Step 2: Train the model
				model, X_test, y_test = train_temperature_model(data)

				# Step 3: Evaluate the model and represent the knowledge
				model_knowledge = evaluate_model_and_represent_knowledge(model, X_test, y_test)

				# Step 4: Save the results to a JSON file
				with open(result_file_path, 'w', encoding='utf-8') as f:
					json.dump(model_knowledge, f, ensure_ascii=False, indent=4)
				print(f"Model results updated and saved to: {result_file_path}")

			time.sleep(update_interval_sec)

		except Exception as e:
			print(f"An error occurred: {e}")
			time.sleep(60)


if __name__ == "__main__":
	data_file_path = "climate_change_data.csv"
	result_file_path = "climate_change_model_results.json"

	# Ensure data file exists
	if not os.path.exists(data_file_path):
		print(f"Data file {data_file_path} not found. Please provide the initial data.")
	else:
		update_model_periodically(data_file_path, result_file_path)
