def reverse():
    sentence = "Hello, how are you?"
    # Dictionary comprehension to map words to their reverse
    reverse_words = {word: word[::-1] for word in sentence.split()}
    print(reverse_words)
def main():
    reverse()
if __name__=='__main__':
    main()