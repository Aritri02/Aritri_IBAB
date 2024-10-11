def decimal_to_binary(n):
    output=""
    count=0
    while(n>0):
        r=n%2
        n=n//2
        output=str(r)+output
    for i in output:
       if (i=='1'):
           count=count+1
    return (count)
def main():
    n = int(input("Enter the decimal number: "))
    result = decimal_to_binary(n)
    print("The no of 1s in the decimal form of ", n, "is: ", result)
if __name__ == "__main__":
    main()