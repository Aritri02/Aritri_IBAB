def half_string(s):
   first_half=s[:len(s)//2]
   return (first_half)
def main():
        s = input("Enter the string: ")
        result = half_string(s)
        print("The halved string is: ", result)
if __name__ == "__main__":
    main()