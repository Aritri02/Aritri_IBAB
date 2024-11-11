def lfunction():
    strings = ["apple", "banana", "cherry", "date", "kiwi"]
    sorted_strings = sorted(strings, key=lambda x: x[-1])
    print(sorted_strings)
def main():
    lfunction()
if __name__=='__main__':
    main()