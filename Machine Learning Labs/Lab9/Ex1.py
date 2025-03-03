import pandas as pd
import numpy as np
data = pd.read_csv('/home/ibab/Downloads/simulated_data_multiple_linear_regression_for_ML.csv')
def partition_data(df, threshold):
    # Partition the data based on the threshold
    partitioned_lower = df[df['BP'] <= threshold]  # BP <= threshold
    partitioned_upper = df[df['BP'] > threshold]  # BP > threshold
    return partitioned_lower, partitioned_upper
# Define the thresholds to partition the data
thresholds = [78, 80, 82]
for t in thresholds:
    lower, upper = partition_data(data, t)
    print(f"\nPartitioned data for threshold t = {t}:")
    print(f"Lower partition (BP <= {t}): {len(lower)} samples")
    print(f"Upper partition (BP > {t}): {len(upper)} samples")
print("\nExample of partitioned data (t = 80):")
print(data[data['BP'] <= 80].head())  # Lower partition (BP <= 80)
print(data[data['BP'] > 80].head())   # Upper partition (BP > 80)
