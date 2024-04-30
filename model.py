import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        # print(x.shape)
        return x


class Yolov1(nn.Module):
    def __init__(self, in_channels=3):
        super(Yolov1, self).__init__()
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers()
        self.fc_layer = self._create_fc_layers()

    def forward(self, x):
        x = self.darknet(x)
        # print(x.shape)
        x = self.fc_layer(x)

        return x

    # https://github.com/tanjeffreyz/yolo-v1/blob/main/train.py 참조
    def _create_conv_layers(self):
        in_channels = self.in_channels
        layers = []

        layers += [
            # # Block 1
            # CNNBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            # nn.MaxPool2d(2, 2),
            #
            # # Block 2
            # CNNBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(2, 2),
            #
            # # Block 3
            # CNNBlock(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            # CNNBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # CNNBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            # CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(2, 2),
            #
            # # Block4
            # CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            # CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            # CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            # CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            # CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0),
            # CNNBlock(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            # CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(2, 2),
            #
            # # Block 5
            # CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            # CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            # CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            #
            # # Yolov1 Head Starts
            # CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            #
            # # Block 6
            # CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)


            # Blcok 1
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2,2),

            # Block 5
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),

            # Block 6
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
        ]

        return nn.Sequential(*layers)

    def _create_fc_layers(self):
        fc_layer = [
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, 7 * 7 * (20 + (5 * 2))),
        ]

        return nn.Sequential(*fc_layer)

def test():
    model = Yolov1(in_channels=3)
    x=torch.randn((2, 3, 448, 448))
    print(model(x))

if __name__ == '__main__':
    test() # torch.Size([2, 1470]) -> 2개의 bounding box를 예측하도록 설정했으니 각각에 따른 예측정보가 나옴.


