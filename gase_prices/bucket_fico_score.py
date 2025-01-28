# Convert the data into a DataFrame
from io import StringIO
# Load the data
data = pd.read_csv('gase_prices/Loan_Data.csv')

# Display the first few rows of the dataset
print(data.head())

# Extract FICO scores and default status
fico_scores = df['fico_score']
defaults = df['default']