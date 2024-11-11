def vowels():
    fruits = ['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']
    fruits_with_only_two_vowels = [fruit for fruit in fruits if sum(1 for letter in fruit if letter in 'aeiou') == 2]
    print(fruits_with_only_two_vowels)
def main():
    vowels()
if __name__=='__main__':
    main()

