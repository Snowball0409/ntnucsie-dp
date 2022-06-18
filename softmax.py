import numpy as np
import tensorflow as tf

class softmax:
    def __init__(self):
        print("Create Softmax...")
    
    def Softmax(self,y_true,y_pred):
        L = tf.reduce_sum(y_true-np.log(np.exp(y_pred)/sum(np.exp(y_pred))))
        return 1/(y_true.size)*tf.cast(L,dtype=tf.float32)
    
    # def L_Softmax(self,y_true,y_pred):
    #     L = 
