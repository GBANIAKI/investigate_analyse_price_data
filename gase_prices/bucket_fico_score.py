# Convert the data into a DataFrame
from io import StringIO
import numpy as np

def quantize_fico_scores(fico_scores, num_buckets):
    # Sort FICO scores
    sorted_scores = np.sort(fico_scores)
    
    # Determine bucket boundaries
    bucket_size = len(sorted_scores) // num_buckets
    boundaries = [sorted_scores[i * bucket_size] for i in range(1, num_buckets)]
    
    # Assign each score to a bucket
    buckets = np.digitize(fico_scores, boundaries)
    
    return buckets, boundaries

def calculate_log_likelihood(fico_scores, defaults, boundaries):
    # Assign each score to a bucket
    buckets = np.digitize(fico_scores, boundaries)
    
    log_likelihood = 0
    for bucket in range(len(boundaries) + 1):
        # Get indices of scores in the current bucket
        indices = np.where(buckets == bucket)[0]
        if len(indices) == 0:
            continue
        
        # Calculate ni and ki
        ni = len(indices)
        ki = np.sum(defaults.iloc[indices])
        
        # Calculate pi
        pi = ki / ni if ni > 0 else 0
        
        # Update log-likelihood
        if pi > 0 and pi < 1:
            log_likelihood += ki * np.log(pi) + (ni - ki) * np.log(1 - pi)
    
    return log_likelihood

# Load the data
data = pd.read_csv('gase_prices/Loan_Data.csv')

# Display the first few rows of the dataset
print(data.head())

# Extract FICO scores and default status
fico_scores = df['fico_score']
defaults = df['default']

# Quantize into 5 buckets
num_buckets = 5
buckets, boundaries = quantize_fico_scores(fico_scores, num_buckets)

# Add the bucket information to the DataFrame
df['fico_bucket'] = buckets

print(df[['fico_score', 'fico_bucket']])
# Calculate initial log-likelihood
initial_log_likelihood = calculate_log_likelihood(fico_scores, defaults, boundaries)
print(f"Initial Log-Likelihood: {initial_log_likelihood}")