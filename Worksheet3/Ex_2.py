from random import randint
def max_min(L):
    max=L[0]
    min=L[0]
    for i in range (len(L)):
        if L[i]>max:
            max=L[i]

        if L[i]<min:
            min=L[i]
    print(f"The maximum element is: {max}")
    print(f"The minimum element is: {min}")
def main():
        n=int(input("Enter the size: "))
        L = [randint(120,400) for i in range (n)]
        print(L)
        max_min(L)
if __name__ == "__main__":
    main()