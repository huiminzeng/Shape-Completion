import torch
import torch.nn as nn
import torch.nn.functional as F

############################################################################################3
#   LOG:
#   1. didn't take the second channel of the input data

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #ENCODER:
        self.conv1 = nn.Sequential(         # 1. sandwich
            nn.Conv3d(
                in_channels=1,              # 'InputPlane' of torch 2
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
            nn.BatchNorm3d(160),            # BatchNorm
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
            nn.BatchNorm3d(320),            # BatchNorm
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
            nn.BatchNorm3d(640),            # BatchNorm
            nn.LeakyReLU(0.2),              # activation LeakyReLU SLOPE 0.2 AS ORINGINAL DEFAULT VALUE
        )

        

        self.FC1 = nn.Sequential(           # 5. sandwich
            nn.Linear(640,640),         # FC
            nn.ReLU(),                      # activation ReLU AS ORIGINAL DEFAULT
        )

        self.FC2 = nn.Sequential(           # 6. sandwich
            nn.Linear(640,640),         # FC
            nn.ReLU(),                      # activation ReLU AS ORIGINAL DEFAULT
        )


        #DECODER:
        self.transconv1 = nn.Sequential(    # 1. upsampling
            nn.ConvTranspose3d(             # volumetric full convolution  
                in_channels=2*640,          # 'InputPlane' 640
                out_channels=320,         # 'OutputPlane' 320
                kernel_size=4,              # kernel size 4*4*4
                stride=1,                   # stride 1*1*1
                padding=0,                  # padding 0*0*0
            ),
            nn.BatchNorm3d(320),          # BatchNorm
            nn.ReLU(),                      # activation ReLU
        )

        self.transconv2 = nn.Sequential(    # 2. upsampling
            nn.ConvTranspose3d(             # volumetric full convolution  
                in_channels=2*320,          # 'InputPlane' 320
                out_channels=160,         # 'OutputPlane' 160
                kernel_size=4,              # kernel size 4*4*4
                stride=2,                   # stride 2*2*2
                padding=1,                  # padding 1*1*1
            ),
            nn.BatchNorm3d(160),            # BatchNorm
            nn.ReLU(),                      # activation ReLU
        )

        self.transconv3 = nn.Sequential(    # 3. upsampling
            nn.ConvTranspose3d(             # volumetric full convolution  
                in_channels=2*160,            # 'InputPlane' 160
                out_channels=80,            # 'OutputPlane' 80
                kernel_size=4,              # kernel size 4*4*4
                stride=2,                   # stride 2*2*2
                padding=1,                  # padding 1*1*1
            ),
            nn.BatchNorm3d(80),             # BatchNorm
            nn.ReLU(),                      # activation ReLU
        )

        self.transconv4 = nn.Sequential(    # 4. upsampling
            nn.ConvTranspose3d(             # volumetric full convolution  
                in_channels=2*80,             # 'InputPlane' 160
                out_channels=1,             # 'OutputPlane' 1
                kernel_size=4,              # kernel size 4*4*4
                stride=2,                   # stride 2*2*2
                padding=1,                  # padding 1*1*1
            ),
        )

    def forward(self, x):                 
        enc1 = self.conv1(x)
        #print('conv1 ok, shape: ', enc1.shape)  #outout size 80 * 16^3
        enc2 = self.conv2(enc1)
        #print('conv2 ok, shape: ', enc2.shape)  #outout size 160 * 8^3
        enc3 = self.conv3(enc2)
        #print('conv3 ok, shape: ', enc3.shape)  #outout size 320 * 4^3
        enc4 = self.conv4(enc3)
        #print('conv4 ok, shape: ', enc4.shape)  #outout size 640 * 1^3
        encoded = enc4

        bottleneck = encoded.view(encoded.shape[0],-1)
        #print(bottleneck.shape)                 #output size 16 * 640

        bottleneck = self.FC1(bottleneck)
        #print(bottleneck.shape)                 #output size 16 * 640

        bottleneck = self.FC2(bottleneck)
        #print(bottleneck.shape)                 #output size 16 * 640

        bottlenecked = bottleneck.view(-1,640,1,1,1)
        #print(bottlenecked.shape)

        #################################################################
        #################### NO NOISE YET! ##############################
        #################################################################

        d1 = torch.cat((bottlenecked,enc4),dim=1)   #dim wrong!!!!!!!!!!!!!!!!
        #print(d1.shape)                             #output size 1280 * 1^3

        upsamp1 = self.transconv1(d1)
        #print('transposeconv1 ok, shape:', upsamp1.shape)   #output size 320 * 4^3

        d2 = torch.cat((upsamp1,enc3),dim=1)                #output size 640 * 4^3
        #print(d2.shape)

        upsamp2 = self.transconv2(d2)
        #print('transposeconv2 ok, shape:', upsamp2.shape)   #output size 160 * 8^3

        d3 = torch.cat((upsamp2,enc2),dim=1)
        #print(d3.shape)                                     #output size 320 * 8^3

        upsamp3 = self.transconv3(d3)
        #print('transposeconv3 ok, shape:', upsamp3.shape) 

        d4 = torch.cat((upsamp3,enc1),dim=1)
        #print(d4.shape)

        decoded = self.transconv4(d4)
        #print('transposeconv4 ok, shape:', decoded.shape) 

        decoded[:, ].abs_()
        decoded[:].add_(1).log_()
        #print("mmp")

        return decoded


class Discriminator(nn.Module):
    def __init__(self,):
        super(Discriminator, self).__init__()

        #ENCODER:
        self.conv1 = nn.Sequential(         # 1. sandwich
            nn.Conv3d(
                in_channels=2,              # 'InputPlane' of torch 1
                out_channels=80,            # 'OutputPlane' of torch 80 
                kernel_size=4,              # kernel size 4*4*4
                stride=2,                   # kernel step 2*2*2
                padding=1,                  # padding 1*1*1
            ),   
            nn.BatchNorm3d(80),             # BatchNorm
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
            nn.BatchNorm3d(160),            # BatchNorm
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
            nn.BatchNorm3d(320),            # BatchNorm
            nn.LeakyReLU(0.2),              # activation LeakyReLU SLOPE 0.2 AS ORINGINAL DEFAULT VALUE
        )

        self.conv4 = nn.Sequential(         # 4. sandwich
            nn.Conv3d(
                in_channels=320,            # 'InputPlane' of torch 320
                out_channels=1,           # 'OutputPlane' of torch 1
                kernel_size=4,              # kernel size 4*4*4
                stride=1,                   # kernel step 1*1*1
                padding=0,                  # padding 0*0*0
            ),   
            nn.LeakyReLU(0.2),              # activation LeakyReLU SLOPE 0.2 AS ORINGINAL DEFAULT VALUE
        )

    def forward(self, condition, inputs):
        #print("condition shape: ", condition.shape)
        #print("inputs shape: ", inputs.shape)
        dis_in = torch.cat([condition, inputs], dim=1)              
        enc1 = self.conv1(dis_in)
        #print('conv1 ok, shape: ', enc1.shape)  #outout size 80 * 16^3
        enc2 = self.conv2(enc1)
        #print('conv2 ok, shape: ', enc2.shape)  #outout size 160 * 8^3
        enc3 = self.conv3(enc2)
        #print('conv3 ok, shape: ', enc3.shape)  #outout size 320 * 4^3
        enc4 = self.conv4(enc3)
        #print('discriminator conv4 ok, shape: ', enc4.shape)  #outout size 1 * 1^3
        scores = torch.sigmoid(enc4).view(-1)
        #print('discriminator shape: ', scores.shape)
        #bottleneck = encoded.view(encoded.shape[0],-1)
        #print(bottleneck.shape)                 #output size 16 * 640


        return scores