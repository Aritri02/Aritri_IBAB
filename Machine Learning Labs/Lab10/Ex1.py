import math
from collections import Counter
def calculate_entropy(data):
    # Count the occurrences of each class
    class_counts = Counter(data)
    # Total number of data points
    total_points = len(data)
    # Calculate the entropy
    entropy = 0
    for count in class_counts.values():
        probability = count / total_points  # Probability of each class
        entropy -= probability * math.log2(probability)  # Entropy formula
    return entropy
# Example usage
data = ['class_1', 'class_1', 'class_2', 'class_2', 'class_3', 'class_1', 'class_3', 'class_3', 'class_2']
entropy_value = calculate_entropy(data)
print(f"Entropy: {entropy_value}")


# import numpy as np
# from math import log,e
# def entropy4(labels, base=None):
#   value,counts = np.unique(labels, return_counts=True)
#   norm_counts = counts / counts.sum()
#   base = e if base is None else base
#   return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()
# labels = [1,3,5,2,3,5,3,2,1,3,4,5]
# print(entropy4(labels))

