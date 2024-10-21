def max_min(scores):
    out=list(scores.values())
    max_value = out[0]
    min_value = out[0]
    for j in out:
            if j>max_value:
                max_value=j
            elif j<min_value:
                min_value=j
    return max_value,min_value
def main():
    scores = {"Rahul": 80, "Shubham": 120, "Virat": 100}
    max,min = max_min(scores)
    print(f'The maximum value in the dictionary is: {max} and the minimum value is: {min}')
if __name__ == "__main__":
    main()