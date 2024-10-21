def sum_dictionary(scores):
    # out=[]
    # for i in scores.values():
    #     out.append(i)
    # # return out
    # add=0
    # for j in out:
    #     add=add+j
    # return add
    add=sum(scores.values())
    return add
def main():
        scores = {"Rahul": 80, "Shubham": 120, "Virat": 100}
        result = sum_dictionary(scores)
        print(f'The sum of all the values in the dictionary is: {result}')
if __name__ == "__main__":
    main()