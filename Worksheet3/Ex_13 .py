def unique(dict):
    max_count=0
    result_key=None
    for key,val in dict.items():
        unique_val=set(val)
        unique_len=len(unique_val)
        if unique_len>max_count:
            max_count=unique_len
            result_key=key
    return result_key
def main():
        dict ={"Gfg" : [5, 7, 7, 7, 7], "is" : [6, 7, 7, 7], "Best" : [9, 9, 6, 5, 5]}
        print(unique(dict))
if __name__ == "__main__":
    main()