import torch
import torch.nn as nn
import math

def BNFunc(*args, **kwargs):
    #if torch.distributed._initialized:
    #    return SyncBatchNorm2d(*args, **kwargs, group_size=1, group=None, sync_stats=True)
    if True:
        return nn.BatchNorm2d(*args, **kwargs)
    else:
        return nn.BatchNorm2d(*args, **kwargs)

BN=BNFunc

class SimpleNet(nn.Module):#only five layers
    def __init__(self):
        super(SimpleNet, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),#112
            nn.ReLU(True),
            BN(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),#56
            nn.ReLU(True),
            BN(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),#28
            BN(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),#14
            BN(256),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(10*10*256, 2)
        self.classfier = nn.Sigmoid()

        self.weight_init()

    def forward(self, input):
        b,c,h,w=input.size()

        out = self.main(input)
        out = out.view(b,-1)
        out = self.output(out)
        out = self.classfier(out)
        return out


    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias is None
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
