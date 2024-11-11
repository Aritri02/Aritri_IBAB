def lists():
    fruits=['mango', 'kiwi', 'strawberry', 'guava', 'pineapple', 'mandarin orange']
    cap_fruit=[i.capitalize() for i in fruits ]
    print(cap_fruit)
def main():
    lists()
if __name__=='__main__':
    main()