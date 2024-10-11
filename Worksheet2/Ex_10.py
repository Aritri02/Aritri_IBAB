def join_string(s1,s2):
   joined= s1+ " "+s2
   return (joined)
def main():
        s1 = input("Enter the first string: ")
        s2=input("Enter the second string: ")
        result = join_string(s1,s2)
        print("The joined string is: ", result)
if __name__ == "__main__":
    main()