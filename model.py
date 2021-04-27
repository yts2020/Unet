import torch
import torch.nn as nn


class Double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Double_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.double_conv1 = Double_conv(3, 64)
        self.double_conv2 = Double_conv(64, 128)
        self.double_conv3 = Double_conv(128, 256)
        self.double_conv4 = Double_conv(256, 512)
        self.double_conv5 = Double_conv(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_conv6 = Double_conv(1024, 512)
        self.double_conv7 = Double_conv(512, 256)
        self.double_conv8 = Double_conv(256, 128)
        self.double_conv9 = Double_conv(128, 64)
        self.output = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.double_conv1(x)
        pool1 = self.pool(x1)
        x2 = self.double_conv2(pool1)
        pool2 = self.pool(x2)
        x3 = self.double_conv3(pool2)
        pool3 = self.pool(x3)
        x4 = self.double_conv4(pool3)
        pool4 = self.pool(x4)
        x5 = self.double_conv5(pool4)
        up4 = self.upsample4(x5)
        up4 = torch.cat((x4, up4), dim=1)
        up4 = self.double_conv6(up4)
        up3 = self.upsample3(up4)
        up3 = torch.cat((x3, up3), dim=1)
        up3 = self.double_conv7(up3)
        up2 = self.upsample2(up3)
        up2 = torch.cat((x2, up2), dim=1)
        up2 = self.double_conv8(up2)
        up1 = self.upsample1(up2)
        up1 = torch.cat((x1, up1), dim=1)
        up1 = self.double_conv9(up1)
        output = self.output(up1)
        return output
