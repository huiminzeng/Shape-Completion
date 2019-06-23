import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

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
                in_channels=640,          # 'InputPlane' 640
                out_channels=320,         # 'OutputPlane' 320
                kernel_size=4,              # kernel size 4*4*4
                stride=1,                   # stride 1*1*1
                padding=0,                  # padding 0*0*0
            ),
            #nn.BatchNorm3d(320),          # BatchNorm
            nn.ReLU(),                      # activation ReLU
        )

        self.transconv2 = nn.Sequential(    # 2. upsampling
            nn.ConvTranspose3d(             # volumetric full convolution  
                in_channels=320,          # 'InputPlane' 320
                out_channels=160,         # 'OutputPlane' 160
                kernel_size=4,              # kernel size 4*4*4
                stride=2,                   # stride 2*2*2
                padding=1,                  # padding 1*1*1
            ),
            #nn.BatchNorm3d(160),            # BatchNorm
            nn.ReLU(),                      # activation ReLU
        )

        self.transconv3 = nn.Sequential(    # 3. upsampling
            nn.ConvTranspose3d(             # volumetric full convolution  
                in_channels=160,            # 'InputPlane' 160
                out_channels=80,            # 'OutputPlane' 80
                kernel_size=4,              # kernel size 4*4*4
                stride=2,                   # stride 2*2*2
                padding=1,                  # padding 1*1*1
            ),
            #nn.BatchNorm3d(80),             # BatchNorm
            nn.ReLU(),                      # activation ReLU
        )

        self.transconv4 = nn.Sequential(    # 4. upsampling
            nn.ConvTranspose3d(             # volumetric full convolution  
                in_channels=80,             # 'InputPlane' 160
                out_channels=1,             # 'OutputPlane' 1
                kernel_size=4,              # kernel size 4*4*4
                stride=2,                   # stride 2*2*2
                padding=1,                  # padding 1*1*1
            ),
        )

    def forward(self, x):                   # input size  2 * 32^3
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1)
        #print("mmp!", x.shape)

        x = self.FC1(x)
        x = self.FC2(x)
        #print("x shape: ", x.shape)
        x = x.view(-1,640,1,1,1)
        
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        out = self.transconv4(x)
        #print("outputs shape: ", out.shape)
        return out