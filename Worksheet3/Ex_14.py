def remove_duplicates(s):
    dict={}
    word=s.split()
    for i in word:
        dict[i]=None
    return ' '.join(dict.keys())
def main():
        s='I was happy because everybody was happy'
        print(remove_duplicates(s))
if __name__ == "__main__":
    main()