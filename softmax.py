import tensorflow as tf

class softmax:
    def __init__(self):
        print("Create Softmax...")
    
    def Softmax(self,y_true,y_pred):
        L = tf.reduce_mean(y_true-tf.math.log(tf.math.exp(y_pred)/sum(tf.math.exp(y_pred))))
        return tf.cast(L,dtype=tf.float32)
    
