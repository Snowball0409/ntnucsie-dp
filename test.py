from softmax import *
import numpy as np

def main():
    loss = softmax()
    input = np.array([1,2,3,4,5])
    print(loss(input))

main()