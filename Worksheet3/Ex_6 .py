def extract(L,k):
    out=[]
    for i in L:
        if L.count(i)>k:
            out.append(i)
            return out
        else:
            return 0
def main():
        k=int(input('Enter the value: '))
        L=[22,11,67,1,22,45,22,56,101,22,40,220]
        print(extract(L,k))
if __name__ == "__main__":
        main()