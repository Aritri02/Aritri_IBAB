def decimal_to_binary(n):
    output=""
    while (n>0):
        r = n % 2
        n=n//2
        output=str(r)+output
    return(output)
def main():
        n = int(input("Enter the decimal number: "))
        result = decimal_to_binary(n)
        print("The binary form of ",n, "is: ",result)
if __name__ == "__main__":
        main()