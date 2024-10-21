def duplicate_num(L):
    duplicate=[]
    for i in L:
        if L.count(i)==2:
            duplicate.append(i)
    print(f'The list of duplicate numbers is: {duplicate}')
def main():
        n=int(input("Enter the size: "))
        L =[]
        for i in range(0,n):
            element=int(input("Enter the elements of the list: "))
            L.append(element)
        print(L)
        duplicate_num(L)
if __name__ == "__main__":
    main()