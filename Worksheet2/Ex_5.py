def power(b,n):
    p=pow(b,n)
    return(p)
def main():
     b = int(input("Enter the base number: "))
     n=int(input("Enter the power: "))
     result = power(b,n)
     print("The result of base ", b, "raised to the power ",n,"is: ", result)


if __name__ == "__main__":
    main()