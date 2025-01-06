import numpy as np
import matplotlib.pyplot as plt
def math_imple(x):
    y = (2*x*x)+(3*x)+4
    for i in x:
        plt.plot(x, y)
    plt.show()
def main():
    a = np.linspace(-10, 10, 100)
    print(math_imple(a))
if __name__ == '__main__':
    main()