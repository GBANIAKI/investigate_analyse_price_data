import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the data
data = pd.read_csv('gase_prices/Loan_Data.csv')

# Display the first few rows of the dataset
print(data.head())

# Select features and target variable
X = data[['income', 'loan_amt_outstanding', 'total_debt_outstanding', 'years_employed', 'fico_score']]
y = data['default']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba)}")

# Function to calculate expected loss
def calculate_expected_loss(income, loan_amt_outstanding, total_debt_outstanding, years_employed, fico_score, recovery_rate=0.1):
    # Prepare the input data
    input_data = pd.DataFrame([[income, loan_amt_outstanding, total_debt_outstanding, years_employed, fico_score]], 
                              columns=['income', 'loan_amt_outstanding', 'total_debt_outstanding', 'years_employed', 'fico_score'])
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the probability of default
    probability_of_default = model.predict_proba(input_data_scaled)[:, 1][0]
    
    # Calculate expected loss
    expected_loss = probability_of_default* (1 - recovery_rate) * loan_amt_outstanding
    return expected_loss

# Example usage
income = 50000
loan_amt_outstanding = 20000
total_debt_outstanding = 30000
years_employed = 5
fico_score = 650

expected_loss = calculate_expected_loss(income, loan_amt_outstanding, total_debt_outstanding, years_employed, fico_score)
print(f"Expected Loss: {expected_loss}")