# Redefining the function and variables after execution state reset

import numpy as np
import math

def Transform(x1, x2):
    """Transform function to map (x1, x2) into higher-dimensional space."""
    return np.array([ x1 ** 2 , (math.sqrt(2) * x1 * x2) , x2 ** 2])


def main():
    # Given vectors
    x1 = np.array([3, 6])
    x2 = np.array([10, 10])

    # Transform the vectors
    phi_x1 = Transform(x1[0], x2[0])
    phi_x2 = Transform(x1[1], x2[1])
    print("The transformed x1 is: ",phi_x1)
    print("The transformed x2 is: ", phi_x2)

    # Compute the dot product in higher dimension
    dot_product = np.dot(phi_x1, phi_x2)

    # Print the result
    print("The dot product in higher dimension is: ",dot_product)

if __name__ == "__main__":
    main()
