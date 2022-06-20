from softmax import *
import numpy as np

def main():
    loss = softmax()
    input = np.array([0.0,0.0,0.0,0.1,0.9])
    true = np.array([0,0,0,0,1])
    print(loss.Softmax(true,input))

main()