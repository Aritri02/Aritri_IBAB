def first_occurence(s):
    for i in range(len(s)):
        if (s[i]=='c'):
            return i
    return -1
def main():
        s = input("Enter the string: ")
        result = first_occurence(s)
        if result !=-1:
         print("The first occurence of c in the string is at index: ", result)
        else:
            print("The character c is not found")
if __name__ == "__main__":
    main()