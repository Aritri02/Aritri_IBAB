import numpy as np
def matrix():
    A=np.array([[1, 2, 3], [4, 5, 6]])
    print(A)
    At=A.transpose()
    print(At)
    r=np.dot(A,At)
    print(r)
def main():
    matrix()
if __name__ == '__main__':
    main()
