import torch
import torch.nn as nn


class VUnet(nn.Module):
    def __init__(self, pretrained_path=None):
        super(VUnet, self).__init__()
        self.d1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.d2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.d3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.d4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.d5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.d5_transpose = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.u1 = up_conv_block([512, 512], [512, 512], [3, 3, 3], [1, 1, 1], 512, 256, 2, 2)
        self.u2 = up_conv_block([256, 256], [256, 256], [3, 3, 3], [1, 1, 1], 256, 128, 2, 2)
        self.u3 = up_conv_block([128, 128], [128, 128], [3, 3], [1, 1], 128, 64, 2, 2)
        self.u4 = up_conv_block([64, 64], [64, 64], [3, 3], [1, 1], 0, 0, 2, 2)
        self.final = nn.Sequential(nn.Conv2d(64, 1, kernel_size=(1, 1)), nn.Sigmoid())
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        '''
        # The following code is for testing the pretrained classification functionalities.

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )
        '''
        if pretrained_path is not None:
            self._load_parameters(pretrained_path)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.mp(x1))
        x3 = self.d3(self.mp(x2))
        x4 = self.d4(self.mp(x3))
        x5 = self.d5_transpose(self.d5(self.mp(x4)))
        x6 = self.u1(x5 + x4)
        x7 = self.u2(x6 + x3)
        x8 = self.u3(x7 + x2)
        x9 = self.u4(x8 + x1)
        output = self.final(x9)
        return output

    # the following is for pretrained classification testing.
    '''
    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    '''
    def _load_parameters(self, pretrained_path):
        own_state = self.state_dict()
        pretrained_state = torch.load(pretrained_path)
        for name, param in pretrained_state.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)

    def freeze(self):
        for param in self.d1.parameters():
            param.requires_grad = False
        for param in self.d2.parameters():
            param.requires_grad = False
        for param in self.d3.parameters():
            param.requires_grad = False
        for param in self.d4.parameters():
            param.requires_grad = False
        for param in self.d5.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.d1.parameters():
            param.requires_grad = True
        for param in self.d2.parameters():
            param.requires_grad = True
        for param in self.d3.parameters():
            param.requires_grad = True
        for param in self.d4.parameters():
            param.requires_grad = True
        for param in self.d5.parameters():
            param.requires_grad = True


def conv_layer(ch_in, ch_out, k_size, p_size):  # A single convolutional layer with ReLU
    layer = nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=p_size),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], (k_list[i], k_list[i]), p_list[i]) for i in range(len(in_list))]
    # layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def up_conv_block(in_list, out_list, k_list, p_list, transpose_ch_in, transpose_ch_out, transpose_k, transpose_s):
    layers = [conv_layer(in_list[i], out_list[i], (k_list[i], k_list[i]), p_list[i]) for i in range(len(in_list))]
    if transpose_ch_in != 0 and transpose_ch_out != 0:
        layers += [nn.ConvTranspose2d(in_channels=transpose_ch_in, out_channels=transpose_ch_out,
                                      kernel_size=transpose_k, stride=transpose_s)]
    return nn.Sequential(*layers)
