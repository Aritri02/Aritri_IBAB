def dictionary(dict):
    keys=[]
    values=[]
    for i,j in dict.items():
        keys.append(i)
        values.append(j)
    return keys,values
def main():
        dict={'Alan':1,'Aritri':2,'Arushi':3,'Nandi':4,'Khusbu':5}
        k,v=dictionary(dict)
        print(f'The values are {v} and the keys are {k}')
if __name__ == "__main__":
    main()