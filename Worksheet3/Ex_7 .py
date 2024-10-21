def remove_occurences(L,element):
    out = [i for i in L if i != element]
    return out
    # c=L.count(element)
    # for i in range(c):
    #     L.remove(element)
    # return L
def main():
        element=int(input('Enter the element you want to remove: '))
        L = [22, 11, 67, 1, 22, 45, 22, 56, 101, 22, 40, 220]
        print(f'List before removal of {element}: {L}')
        result=remove_occurences(L,element)
        print(f'List after removal of {element}: {result}')
if __name__ == "__main__":
    main()
