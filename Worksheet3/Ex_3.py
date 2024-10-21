from random import randint
def even_num(L):
    even=[]
    for i in L:
        if i%2==0:
            even.append(i)
    print(f'The list of even numbers is: {even}')
def main():
        n=int(input("Enter the size: "))
        L = [randint(120,400) for i in range (n)]
        print(L)
        even_num(L)
if __name__ == "__main__":
    main()