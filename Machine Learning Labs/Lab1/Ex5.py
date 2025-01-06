import numpy as np
import matplotlib.pyplot as plt
def math_imple(x):
    y = x**2
    y1= 2*x
    for i in x:
        plt.plot(x, y,marker='*',mfc='r')
        plt.plot(x, y1,marker='*',mfc='r')
    plt.show()
def main():
    a = np.linspace(-100, 100, 100)
    print(math_imple(a))
if __name__ == '__main__':
    main()