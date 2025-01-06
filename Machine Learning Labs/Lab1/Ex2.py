import matplotlib.pyplot as plt
import numpy as np
def math_plot(x):
    y=(2*x)+3
    for i in x:
        plt.plot(x,y)
    plt.show()
def main():
    a=np.linspace(-100,100,100)
    print(math_plot(a))
if __name__ == '__main__':
    main()

