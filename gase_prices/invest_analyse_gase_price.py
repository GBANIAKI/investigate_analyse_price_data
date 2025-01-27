# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import warnings
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Data Loading and Preprocessing

# Define the data as a multi-line string
data = """
Dates,Prices
10/31/20,1.01E+01
11/30/20,1.03E+01
12/31/20,1.10E+01
1/31/21,1.09E+01
2/28/21,1.09E+01
3/31/21,1.09E+01
4/30/21,1.04E+01
5/31/21,9.84E+00
6/30/21,1.00E+01
7/31/21,1.01E+01
8/31/21,1.03E+01
9/30/21,1.02E+01
10/31/21,1.01E+01
11/30/21,1.12E+01
12/31/21,1.14E+01
1/31/22,1.15E+01
2/28/22,1.18E+01
3/31/22,1.15E+01
4/30/22,1.07E+01
5/31/22,1.07E+01
6/30/22,1.04E+01
7/31/22,1.05E+01
8/31/22,1.04E+01
9/30/22,1.08E+01
10/31/22,1.10E+01
11/30/22,1.16E+01
12/31/22,1.16E+01
1/31/23,1.21E+01
2/28/23,1.17E+01
3/31/23,1.20E+01
4/30/23,1.15E+01
5/31/23,1.12E+01
6/30/23,1.09E+01
7/31/23,1.14E+01
8/31/23,1.11E+01
9/30/23,1.15E+01
10/31/23,1.18E+01
11/30/23,1.22E+01
12/31/23,1.28E+01
1/31/24,1.26E+01
2/29/24,1.24E+01
3/31/24,1.27E+01
4/30/24,1.21E+01
5/31/24,1.14E+01
6/30/24,1.15E+01
7/31/24,1.16E+01
8/31/24,1.15E+01
9/30/24,1.18E+01
"""

# Read the data into a pandas DataFrame
df = pd.read_csv(StringIO(data), parse_dates=['Dates'], dayfirst=False)

# Convert 'Prices' from scientific notation to float
df['Prices'] = df['Prices'].astype(float)

# Set 'Dates' as the index
df.set_index('Dates', inplace=True)

# Sort the index to ensure chronological order
df.sort_index(inplace=True)

# Display the first few rows
print("Data Snapshot:")
print(df.head())

# 2. Feature Engineering

# Reset index to access 'Dates' as a column
df_reset = df.reset_index()

# Create additional time-based features
df_reset['Year'] = df_reset['Dates'].dt.year
df_reset['Month'] = df_reset['Dates'].dt.month
df_reset['Day'] = df_reset['Dates'].dt.day

# Create rolling statistics as features
df_reset['Price_Lag_1'] = df_reset['Prices'].shift(1)
df_reset['Price_Lag_2'] = df_reset['Prices'].shift(2)
df_reset['Price_Lag_3'] = df_reset['Prices'].shift(3)
df_reset['Rolling_Mean_3'] = df_reset['Prices'].rolling(window=3).mean()
df_reset['Rolling_STD_3'] = df_reset['Prices'].rolling(window=3).std()

# Encode 'Month' as cyclical features to capture seasonality
df_reset['Month_Sin'] = np.sin(2 * np.pi * df_reset['Month']/12)
df_reset['Month_Cos'] = np.cos(2 * np.pi * df_reset['Month']/12)

# Drop rows with NaN values resulting from lag features
df_model = df_reset.dropna().copy()

# Define feature columns
feature_cols = ['Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3',
                'Rolling_Mean_3', 'Rolling_STD_3',
                'Month_Sin', 'Month_Cos']

# Define X and y
X = df_model[feature_cols]
y = df_model['Prices']

# 3. Train-Test Split

# Define the split point (e.g., last 12 months for testing)
split_point = -12
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)

# 4. Modeling

# 4.1. SARIMA Model

# For SARIMA, we typically do not use exogenous variables in this context
sarima_order = (1, 1, 1)
sarima_seasonal_order = (1, 1, 1, 12)  # Seasonal component with period=12

sarima_model = SARIMAX(y_train, order=sarima_order, seasonal_order=sarima_seasonal_order)
sarima_result = sarima_model.fit(disp=False)

# Forecast
sarima_forecast = sarima_result.predict(start=y_test.index[0], end=y_test.index[-1], dynamic=False)

# 4.2. Prophet Model

# Prophet requires the dataframe to have columns 'ds' and 'y'
prophet_train = df_reset[['Dates', 'Prices']].dropna().copy()
prophet_train = prophet_train.rename(columns={'Dates': 'ds', 'Prices': 'y'})

prophet_model = Prophet(yearly_seasonality=True, seasonal_mode='additive')
prophet_model.fit(prophet_train)

# Create a dataframe for future dates (testing dates)
prophet_future = prophet_train.tail(12)[['ds']].copy()

# Forecast
prophet_forecast = prophet_model.predict(prophet_future)
prophet_pred = prophet_forecast['yhat'].values

# 4.3. Gradient Boosting Regressor (XGBoost)

# Initialize the model
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model
gbr_model.fit(X_train, y_train)

# Predict
gbr_pred = gbr_model.predict(X_test)

# 5. Evaluation

# Define a helper function to compute evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse

print("\nModel Evaluation:")
sarima_mae, sarima_rmse = evaluate_model(y_test, sarima_forecast, "SARIMA")
prophet_mae, prophet_rmse = evaluate_model(y_test, prophet_pred, "Prophet")
gbr_mae, gbr_rmse = evaluate_model(y_test, gbr_pred, "Gradient Boosting Regressor")

# 6. Model Comparison

# Compare models based on RMSE
models = ['SARIMA', 'Prophet', 'Gradient Boosting']
rmse_scores = [sarima_rmse, prophet_rmse, gbr_rmse]

comparison_df = pd.DataFrame({'Model': models, 'RMSE': rmse_scores})
print("\nModel Comparison:")
print(comparison_df.sort_values('RMSE'))

# 7. Residual Analysis for the Best Model

# Assuming Gradient Boosting has the lowest RMSE
best_model_name = comparison_df.sort_values('RMSE').iloc[0]['Model']
print(f"\nBest Model: {best_model_name}")

if best_model_name == 'SARIMA':
    residuals = y_test - sarima_forecast
elif best_model_name == 'Prophet':
    residuals = y_test.values - prophet_pred
else:
    residuals = y_test.values - gbr_pred

# Plot residuals
plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True)
plt.title(f'Residuals Distribution for {best_model_name}')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(x=y_test.index, y=residuals)
plt.title(f'Residuals Over Time for {best_model_name}')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# 8. Enhanced Price Estimation Function Using the Best Model

def enhanced_estimate_price(input_date, model='gbr'):
    """
    Estimate the natural gas price for a given date using the best model.
    
    Parameters:
    input_date (str): Date string in 'MM/DD/YY' or 'MM/DD/YYYY' format.
    model (str): Model to use for prediction ('sarima', 'prophet', 'gbr').
    
    Returns:
    float: Estimated natural gas price.
    """
    # Parse the input date
    try:
        date = pd.to_datetime(input_date)
    except Exception as e:
        raise ValueError("Date format should be MM/DD/YY or MM/DD/YYYY") from e
    
    # Check if the date is within the training data
    last_train_date = df.index[split_point -1]
    if date <= last_train_date:
        # Retrieve the actual price if available
        if date in df.index:
            return df.loc[date, 'Prices']
        else:
            return "Date is within the training period but not available in the dataset."
    else:
        # Forecast based on the selected model
        if model.lower() == 'gbr':
            # Generate features for the new date
            # For simplicity, using the last available data for lag features
            last_available = df_reset.iloc[-1]
            new_entry = {
                'Price_Lag_1': last_available['Prices'],
                'Price_Lag_2': last_available['Price_Lag_1'],
                'Price_Lag_3': last_available['Price_Lag_2'],
                'Rolling_Mean_3': (last_available['Prices'] + last_available['Price_Lag_1'] + last_available['Price_Lag_2']) / 3,
                'Rolling_STD_3': df_reset['Prices'].rolling(window=3).std().iloc[-1],
                'Month_Sin': np.sin(2 * np.pi * date.month / 12),
                'Month_Cos': np.cos(2 * np.pi * date.month / 12)
            }
            X_new = pd.DataFrame([new_entry])
            predicted_price = gbr_model.predict(X_new)[0]
            return round(predicted_price, 2)
        elif model.lower() == 'sarima':
            # Extend the SARIMA model forecast
            steps = (date.year - y_test.index[-1].year) * 12 + (date.month - y_test.index[-1].month)
            if steps < 1 or steps > 12:
                return "Can only forecast up to 12 months beyond the dataset."
            sarima_future = sarima_result.get_forecast(steps=steps)
            predicted_price = sarima_future.predicted_mean[-1]
            return round(predicted_price, 2)
        elif model.lower() == 'prophet':
            # Use Prophet to forecast the specific date
            future = pd.DataFrame({'ds': [date]})
            forecast = prophet_model.predict(future)
            predicted_price = forecast['yhat'].values[0]
            return round(predicted_price, 2)
        else:
            raise ValueError("Invalid model selected. Choose from 'sarima', 'prophet', or 'gbr'.")

# 9. Extrapolation Example Using the Best Model (Gradient Boosting Regressor)

# Example dates within the dataset
print("\nPrice Estimates:")
print("Price on 05/31/21:", enhanced_estimate_price('05/31/21'))
print("Price on 10/31/23:", enhanced_estimate_price('10/31/23'))

# Example dates for extrapolation (up to one year beyond the latest date in data)
print("Estimated Price on 10/31/24:", enhanced_estimate_price('10/31/24'))
print("Estimated Price on 03/15/25:", enhanced_estimate_price('03/15/25', model='sarima'))  # Using SARIMA
print("Estimated Price on 03/15/25:", enhanced_estimate_price('03/15/25', model='prophet'))  # Using Prophet

# 10. Visualization of Predictions

# Plot actual vs predicted for testing set
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Prices', marker='o')
plt.plot(y_test.index, sarima_forecast, label='SARIMA Forecast', marker='x')
plt.plot(y_test.index, prophet_pred, label='Prophet Forecast', marker='s')
plt.plot(y_test.index, gbr_pred, label='Gradient Boosting Forecast', marker='^')
plt.title('Actual vs Predicted Natural Gas Prices')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Plot future predictions using the best model (Gradient Boosting Regressor)

# Generate future dates for one year extrapolation
future_months = 12
last_date = df.index[-1]
future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=future_months, freq='M')

# Create a DataFrame for future dates
future_features = []

for date in future_dates:
    # Retrieve the last 3 prices
    price_lag_1 = df.loc[df.index == last_date, 'Prices'].values[0]
    price_lag_2 = df.loc[df.index == last_date - pd.DateOffset(months=1), 'Prices'].values[0]
    price_lag_3 = df.loc[df.index == last_date - pd.DateOffset(months=2), 'Prices'].values[0]
    rolling_mean_3 = (price_lag_1 + price_lag_2 + price_lag_3) / 3
    rolling_std_3 = df['Prices'].rolling(window=3).std().iloc[-1]
    month_sin = np.sin(2 * np.pi * date.month / 12)
    month_cos = np.cos(2 * np.pi * date.month / 12)
    
    future_features.append({
        'Price_Lag_1': price_lag_1,
        'Price_Lag_2': price_lag_2,
        'Price_Lag_3': price_lag_3,
        'Rolling_Mean_3': rolling_mean_3,
        'Rolling_STD_3': rolling_std_3,
        'Month_Sin': month_sin,
        'Month_Cos': month_cos
    })
    
    # Update for next iteration
    last_date = date

future_df = pd.DataFrame(future_features, index=future_dates)

# Predict future prices using Gradient Boosting Regressor
future_df['Predicted_Prices_GBR'] = gbr_model.predict(future_df[feature_cols])

# For SARIMA and Prophet, we can generate forecasts similarly if needed

# Plot historical and future predicted prices
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Prices'], label='Historical Prices', marker='o')
plt.plot(future_df.index, future_df['Predicted_Prices_GBR'], label='Predicted Prices (GBR)', marker='x', linestyle='--')
plt.title('Historical and Predicted Natural Gas Prices (Gradient Boosting)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# 11. Saving the Models for Future Use (Optional)

import joblib

# Save the Gradient Boosting model
joblib.dump(gbr_model, 'gradient_boosting_model.pkl')

# Save the SARIMA model
sarima_result.save('sarima_model.pkl')

# Save the Prophet model
prophet_model.save('prophet_model.pkl')

print("\nModels have been saved for future use.")