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
def information_gain(parent_data, left_child_data, right_child_data):
    parent_entropy = calculate_entropy(parent_data)
    # Compute the entropy of the left and right children
    left_entropy = calculate_entropy(left_child_data)
    right_entropy = calculate_entropy(right_child_data)
    # Calculate the weighted average entropy of the children
    left_weight = len(left_child_data) / len(parent_data)
    right_weight = len(right_child_data) / len(parent_data)
    weighted_average_entropy = (left_weight * left_entropy) + (right_weight * right_entropy)
    # Information Gain is the parent entropy minus the weighted average entropy of the children
    info_gain = parent_entropy - weighted_average_entropy
    return info_gain

parent_data = ['class_1', 'class_1', 'class_2', 'class_2', 'class_3', 'class_1', 'class_3', 'class_3', 'class_2']
left_child_data = ['class_1', 'class_1', 'class_2', 'class_1']
right_child_data = ['class_2', 'class_3', 'class_3', 'class_3']
ev=calculate_entropy(parent_data)
print(f"The entropy of parent is:{ev}")
info_gain_value = information_gain(parent_data, left_child_data, right_child_data)
print(f"Information Gain: {info_gain_value}")