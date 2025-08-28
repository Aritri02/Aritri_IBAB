import numpy as np
import random

def ReLU(z):
    return np.maximum(0,z)

def softmax(z):
       s = np.sum([np.exp(i) for i in z])
       exp_z=[float(np.exp(j)/s) for j in z]
       return exp_z

def forward_pass(x,w,b):
    a=x

    for j in range(len(w)) :
        z=np.dot(w[j],a)+b[j]
        print(f"\nLayer {j + 1}")
        print("Weights:")
        print(w[j])
        print("Biases:")
        print(b[j])
        print("Z (Weighted sum):")
        print(z)
        if j<(len(w) -1):
            a=ReLU(z)
        else:
            if j == len(w) - 1:
                print(f"1. Type R - if you want to use ReLu activation function at output layer\n2. Type S - to use Soft max activation function at the output layer")
                prompt = input("Enter R/S: ")
                if prompt == "S":
                    a = softmax(z)
                elif prompt == "R":
                    a =ReLU(z)
                else:
                    print("Enter correct prompt")
                    forward_pass(x, w, b)
    return a

def layers(l,input_size):
    neu=[int(input(f"Enter the number of neurons in layer {k+1}: ")) for k in range(l)]
    neu.insert(0,input_size)
    return neu

def weights(n,l):
    w=[]
    b=[]
    choice = input("\nDo you want to manually enter weights and biases? (y/n): ").strip().lower()

    for i in range(l):
        if choice == "y":
            print(f"\nEnter weights for Layer {i + 1} ({n[i + 1]}x{n[i]}):")
            weight = []
            for j in range(n[i + 1]):
                row = []
                for k in range(n[i]):
                    val = float(input(f"Weight[{j}][{k}]: "))
                    row.append(val)
                weight.append(row)
            weight = np.array(weight)

            print(f"\nEnter biases for Layer {i + 1} (size {n[i + 1]}):")
            bias = []
            for j in range(n[i + 1]):
                val = float(input(f"Bias[{j}]: "))
                bias.append(val)
            bias = np.array(bias)
        else:
            weight = np.array([random.uniform(0.001, 1) for _ in range(n[i] * n[i + 1])])
            weight = weight.reshape(n[i + 1], n[i])
            bias = np.array([random.uniform(0.001, 1) for _ in range(n[i + 1])])
        w.append(weight)
        b.append(bias)
    return w,b


def main():
    q=int(input("Enter the number of inputs you want: "))
    x=[float(input(f"Enter the values of each input {i+1}: ")) for i in range(q)]
    l = int(input("Enter the number of layers you want: "))
    x=np.array(x)
    n=layers(l,len(x))
    w,b = weights(n,l)
    output=forward_pass(x,w,b)
    print("Output is: ", output)

if __name__ == "__main__":
        main()









































