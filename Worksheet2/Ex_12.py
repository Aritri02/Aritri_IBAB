def max_frequency(s):
    max_freq=0
    max_char=[]
    for char in s:
        count=s.count(char)
        if count>max_freq:
            max_freq=count
            max_char=[char]
        elif count==max_freq and char not in max_char:
            max_char.append(char)
    return(max_freq,max_char)

def main():
        s = input("Enter the string: ")
        max_freq,max_char = max_frequency(s)
        print("The character(s) with maximum frequency in the string are: ", max_char,"with a frequency of:",max_freq)
if __name__ == "__main__":
    main()