def add(L):
    # total=0
    # for i in range (0,len(L)):
    #     total +=L[i]
    total=sum(L)
    return total
def average(L):
    total=0
    for i in range(0,len(L)):
        total +=L[i]
    avg=total/len(L)
    return(avg)
def main():
        n=int(input("Enter the size of the list: "))
        L = []
        for i in range(0,n):
            element=int(input("Enter the elements of the list: "))
            L.append(element)
        print(L)
        result=add(L)
        print(f'The sum of {n} elements is: {result}')
        result2=average(L)
        print(f'The average of {n} elements is: {result2}')
if __name__ == "__main__":
        main()
