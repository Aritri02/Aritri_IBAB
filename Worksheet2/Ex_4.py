def binary_to_decimal(n):
    i=0
    dn=0
    while(n!=0):
        r=n%10
        dn=dn+(r*pow(2,i))
        #i=i*2
        n=n//10
        i+=1
    return (dn)
def main():
        n = int(input("Enter the binary number: "))
        result = binary_to_decimal(n)
        print("The decimal form of ", n, "is: ", result)
if __name__ == "__main__":
     main()