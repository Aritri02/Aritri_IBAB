def replace_character(s,c1,c2):
    replaced=s.replace(c1,c2)
    return (replaced)
def main():
    s = input("Enter the string: ")
    c1=input("Enter the character to get replaced: ")
    c2=input("Enter the character to replace: ")
    result = replace_character(s,c1,c2)
    print("The replaced string is: ", result)
if __name__ == "__main__":
    main()