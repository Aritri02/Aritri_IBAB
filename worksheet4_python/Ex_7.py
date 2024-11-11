def lfunction():
    arr = [1, -3, 2, -4, -1, 5, 6, -7]
# Sort the array with a lambda function that places negative numbers first
    sorted_arr = sorted(arr, key=lambda x: (x >= 0, abs(x)))
    print(sorted_arr)
def main():
    lfunction()
if __name__=='__main__':
    main()