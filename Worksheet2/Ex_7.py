def digits(n):
    output=''
    while(n!=0):
        d=n%10
        n=n//10
        output=str(d)+","+output
    return (output)
def main():
        n = int(input("Enter the number: "))
        result = digits(n)
        print("The digits of ", n, "are: ", result)
if __name__ == "__main__":
    main()