import numpy as np
import matplotlib.pyplot as plt
def gaussian_pdf(x):
    y = 1/(15*(np.sqrt(2*np.pi)))*(2.71828**(-0.5*((x-0/15)**2)))
    for i in x:
        plt.plot(x, y)
    plt.show()
def main():
    a = np.linspace(-100, 100, 100)
    print(gaussian_pdf(a))
if __name__ == '__main__':
    main()