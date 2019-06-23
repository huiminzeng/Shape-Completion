import torch
import torch.nn as nn
import torch.nn.functional as F

def encoder(inputs, phase_train=True, reuse=False):

    strides    = [1,2,2,2,1]
    with tf.variable_scope("dis"):
        d_1 = tf.nn.conv3d(inputs, weights['wd1'], strides=strides, padding="SAME")
        d_1 = tf.nn.bias_add(d_1, biases['bd1'])
        d_1 = tf.contrib.layers.batch_norm(d_1, is_training=phase_train)                               
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['wd2'], strides=strides, padding="SAME") 
        d_2 = tf.nn.bias_add(d_2, biases['bd2'])
        d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        d_2 = lrelu(d_2, leak_value)
        
        d_3 = tf.nn.conv3d(d_2, weights['wd3'], strides=strides, padding="SAME")  
        d_3 = tf.nn.bias_add(d_3, biases['bd3'])
        d_3 = tf.contrib.layers.batch_norm(d_3, is_training=phase_train)
        d_3 = lrelu(d_3, leak_value) 

        d_4 = tf.nn.conv3d(d_3, weights['wd4'], strides=strides, padding="SAME")     
        d_4 = tf.nn.bias_add(d_4, biases['bd4'])
        d_4 = tf.contrib.layers.batch_norm(d_4, is_training=phase_train)
        d_4 = lrelu(d_4)

        d_5 = tf.nn.conv3d(d_4, weights['wae_d'], strides=[1,1,1,1,1], padding="VALID")     
        d_5 = tf.nn.bias_add(d_5, biases['bae_d'])
        d_5 = tf.nn.sigmoid(d_5)

    print d_5, 'ae5'


def initialiseWeights():

    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 512, 200], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wg5'] = tf.get_variable("wg5", shape=[4, 4, 4, 1, 64], initializer=xavier_init)    

    weights['wd1'] = tf.get_variable("wd1", shape=[4, 4, 4, 1, 64], initializer=xavier_init)
    weights['wd2'] = tf.get_variable("wd2", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wd3'] = tf.get_variable("wd3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wd4'] = tf.get_variable("wd4", shape=[4, 4, 4, 256, 512], initializer=xavier_init)    
    weights['wd5'] = tf.get_variable("wd5", shape=[4, 4, 4, 512, 1], initializer=xavier_init)    
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #ENCODER:
        self.conv1 = nn.Sequential(         # 1. sandwich
            nn.Conv3d(
                in_channels=2,              # 'InputPlane' of torch 2
                out_channels=80,            # 'OutputPlane' of torch 80 
                kernel_size=4,              # kernel size 4*4*4
                stride=2,                   # kernel step 2*2*2
                padding=1,                  # padding 1*1*1
            ),   
            #nn.BatchNorm3d(80),             # BatchNorm
            nn.LeakyReLU(0.2),              # activation LeakyReLU SLOPE 0.2 AS ORINGINAL DEFUALT VALUE
        )

        self.conv2 = nn.Sequential(         # 2. sandwich
            nn.Conv3d(
                in_channels=80,             # 'InputPlane' of torch 80
                out_channels=160,           # 'OutputPlane' of torch 160 
                kernel_size=4,              # kernel size 4*4*4
                stride=2,                   # kernel step 2*2*2
                padding=1,                  # padding 1*1*1
            ),   
            #nn.BatchNorm3d(160),            # BatchNorm
            nn.LeakyReLU(0.2),              # activation LeakyReLU SLOPE 0.2 AS ORINGINAL DEFAULT VALUE
        )

        self.conv3 = nn.Sequential(         # 3. sandwich
            nn.Conv3d(
                in_channels=160,            # 'InputPlane' of torch 160
                out_channels=320,           # 'OutputPlane' of torch 320 
                kernel_size=4,              # kernel size 4*4*4
                stride=2,                   # kernel step 2*2*2
                padding=1,                  # padding 1*1*1
            ),   
            #nn.BatchNorm3d(320),            # BatchNorm
            nn.LeakyReLU(0.2),              # activation LeakyReLU SLOPE 0.2 AS ORINGINAL DEFAULT VALUE
        )

        self.conv4 = nn.Sequential(         # 4. sandwich
            nn.Conv3d(
                in_channels=320,            # 'InputPlane' of torch 320
                out_channels=640,           # 'OutputPlane' of torch 640 
                kernel_size=4,              # kernel size 4*4*4
                stride=1,                   # kernel step 1*1*1
                padding=0,                  # padding 0*0*0
            ),   
            #nn.BatchNorm3d(640),            # BatchNorm
            nn.LeakyReLU(0.2),              # activation LeakyReLU SLOPE 0.2 AS ORINGINAL DEFAULT VALUE
        )
