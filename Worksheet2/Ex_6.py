def prime_check(n):
    if (n==1):
        print("It is not a prime number")
        print("It is a special number")
    for i in range(2,int(n**0.5)+1):
            if (n%i==0):
                print(n,"is not a prime number")
                return False
    return True
def main():
    n = int(input("Enter the number: "))
    if prime_check(n):
        print(n, "is a prime number")
if __name__ == "__main__":
    main()

