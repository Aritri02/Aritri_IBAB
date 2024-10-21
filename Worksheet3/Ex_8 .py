def extract_words(L):
    L=[i.capitalize() for i in L]
    out=[s for s in L if s[0].upper()=='K']
    return out
def main():
        L = ['kite','apple','key','cake','kid','place']
        print(extract_words(L))
if __name__ == "__main__":
    main()