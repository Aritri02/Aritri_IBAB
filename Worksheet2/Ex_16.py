def anagrams(s1,s2):
    s1=s1.lower()
    s2=s2.lower()
    if len(s1)==len(s2):
        sorted_str1=sorted(s1)
        sorted_str2=sorted(s2)
        if sorted_str1==sorted_str2:
            return True
        else:
            return False
def main():
    s1 = input("Enter the first string: ")
    s2 =input("Enter the second string: ")
    if anagrams(s1,s2):
        print(s1," and ", s2, "are anagrams")
    else:
        print(s1, " and ", s2, "are not anagrams")
if __name__ == "__main__":
    main()