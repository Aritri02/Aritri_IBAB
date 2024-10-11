def squares(n):
    sum_of_squares = (n*(n+1) * (2*n+1)) // 6
    return sum_of_squares
def main():
    n=int(input("Enter the number: "))
    result=squares(n)
    print("The sum of squares of first",n,"natural numbers is: ",result)
if __name__=="__main__":
    main()