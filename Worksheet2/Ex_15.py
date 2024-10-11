def count_word(s,w):
    c=s.count(w)
    return(c)
def main():
    s = input("Enter the string: ")
    w=input ("Enter the word you want to count: ")
    times = count_word(s,w)
    print("The word",w,"occurs",times, "times in the string")
if __name__ == "__main__":
    main()