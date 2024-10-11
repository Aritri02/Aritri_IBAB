def alternate_string(s):
   altered=s[::2]
   return (altered)
def main():
        s = input("Enter the string: ")
        result = alternate_string(s)
        print("The halved string is: ", result)
if __name__ == "__main__":
    main()