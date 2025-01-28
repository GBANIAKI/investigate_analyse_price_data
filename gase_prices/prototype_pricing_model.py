import pandas as pd
import numpy as np
from datetime import datetime

# Load the data into a DataFrame
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

# Convert the data into a DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data))

# Explicitly parse dates with a specific format
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')

# Function to calculate the contract value
def calculate_contract_value(injection_dates, withdrawal_dates, injection_rate, withdrawal_rate, max_volume, storage_costs):
    """
    Calculate the value of a gas storage contract.

    Parameters:
    - injection_dates: List of dates when gas is injected.
    - withdrawal_dates: List of dates when gas is withdrawn.
    - injection_rate: Rate at which gas can be injected (units per day).
    - withdrawal_rate: Rate at which gas can be withdrawn (units per day).
    - max_volume: Maximum volume of gas that can be stored.
    - storage_costs: Cost of storing gas per unit per day.

    Returns:
    - The net value of the contract.
    """
    # Initialize variables
    current_volume = 0
    total_value = 0

    # Create a dictionary for quick price lookup
    price_dict = pd.Series(df.Prices.values, index=df.Dates).to_dict()

    # Process injection dates
    for date in injection_dates:
        if date in price_dict:
            # Calculate the amount of gas that can be injected
            inject_amount = min(injection_rate, max_volume - current_volume)
            # Update the current volume
            current_volume += inject_amount
            # Calculate the cost of purchasing the gas
            purchase_cost = inject_amount * price_dict[date]
            # Subtract the purchase cost from the total value
            total_value -= purchase_cost

    # Process withdrawal dates
    for date in withdrawal_dates:
        if date in price_dict:
            # Calculate the amount of gas that can be withdrawn
            withdraw_amount = min(withdrawal_rate, current_volume)
            # Update the current volume
            current_volume -= withdraw_amount
            # Calculate the revenue from selling the gas
            sale_revenue = withdraw_amount * price_dict[date]
            # Add the sale revenue to the total value
            total_value += sale_revenue

    # Calculate storage costs
    total_storage_cost = current_volume * storage_costs * (len(injection_dates) + len(withdrawal_dates))
    # Subtract storage costs from the total value
    total_value -= total_storage_cost

    return total_value

# Test the function with sample inputs
injection_dates = [datetime(2020, 10, 31), datetime(2020, 11, 30)]
withdrawal_dates = [datetime(2021, 1, 31), datetime(2021, 2, 28)]
injection_rate = 100
withdrawal_rate = 100
max_volume = 500
storage_costs = 0.1

contract_value = calculate_contract_value(injection_dates, withdrawal_dates, injection_rate, withdrawal_rate, max_volume, storage_costs)
print(f"The value of the contract is: {contract_value}")