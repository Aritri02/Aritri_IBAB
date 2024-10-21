from random import randint
def insertion_sort(L):
    for j in range(2,len(L)):
        key=L[j]
        i=j-1
        while i>=0 and L[i]> key:
            L[i+1]=L[i]
            i=i-1
        L[i+1]=key
    return L
def main():
    n=int(input('Enter the size: '))
    L=[randint(90,200) for i in range(n)]
    print(L)
    print(insertion_sort(L))
if __name__ == "__main__":
    main()
