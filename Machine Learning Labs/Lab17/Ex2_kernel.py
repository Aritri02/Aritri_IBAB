import numpy as np
import math
from Ex1 import Transform



def polynomial_kernel(a, b):
    """Kernel function that matches the transform function"""
    return (a[0]**2) * (b[0]**2) + 2 * a[0]*b[0]*a[1]*b[1] + (a[1]**2) * (b[1]**2)

def main():
    # Correct input vectors (x1, x2)
    vec1 = np.array([3, 10])
    vec2 = np.array([6, 10])

    # Apply transformation
    phi_vec1 = Transform(vec1[0], vec1[1])
    phi_vec2 = Transform(vec2[0], vec2[1])

    print("The transformed vec1 is:", phi_vec1)
    print("The transformed vec2 is:", phi_vec2)

    # Dot product in transformed space
    dot_product = np.dot(phi_vec1, phi_vec2)
    print("Dot product in transformed space:", dot_product)

    # Apply kernel directly
    kernel_output = polynomial_kernel(vec1, vec2)
    print("Kernel function output:", kernel_output)


if __name__ == "__main__":
    main()
