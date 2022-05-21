import numpy as np

class softmax:
    def __init__(self):
        print("Create Softmax...")
    
    def __call__(self,inputs):
        return np.exp(inputs)/sum(np.exp(inputs))
