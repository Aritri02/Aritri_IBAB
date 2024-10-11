def remove_space(s):
    r=''.join(s.split())
    #r=s.strip()
    return(r)
def main():
    s = input("Enter the string: ")
    result = remove_space(s)
    print("The new string is: ", result)
if __name__ == "__main__":
    main()