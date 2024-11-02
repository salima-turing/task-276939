import schedule
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json


# Assuming you have a function to fetch new data from a source
def fetch_new_climate_data():
    # Implement your data fetching logic here
    # For demonstration, we'll generate random data
    new_data = {
        'Year': [2020, 2021, 2022],
        'Temperature': [31.0, 31.5, 32.0],
        'CO2_Emissions': [400, 405, 410]
    }
    return pd.DataFrame(new_data)


# Existing code remains the same...

def update_model_and_save_results():
    global df, model

    # Fetch new data
    new_data = fetch_new_climate_data()

    # Update the existing data frame with new data
    df = df.append(new_data, ignore_index=True)

    # Retrain the model with the updated data
    model, X_test, y_test = train_temperature_model(df)

    # Evaluate the model and represent knowledge
    model_knowledge = evaluate_model_and_represent_knowledge(model, X_test, y_test)

    # Save the results to a JSON file
    result_file_path = "climate_change_model_results.json"
    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(model_knowledge, f, ensure_ascii=False, indent=4)

    print(f"Model updated and results saved to: {result_file_path}")


if __name__ == "__main__":
    # Set the desired update interval in seconds
    update_interval_seconds = 60 * 60 * 24 * 7  # Update every week (60 seconds * 60 minutes * 24 hours * 7 days)

    # Initial model training and evaluation
    update_model_and_save_results()

    # Schedule the model update
    schedule.every(update_interval_seconds).seconds.do(update_model_and_save_results)

    # Run the scheduler forever
    while True:
        schedule.run_pending()
        time.sleep(1)
