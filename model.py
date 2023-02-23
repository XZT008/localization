from convlstm import ConvLSTM
import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F


class LSG_NET(nn.Module):
    def __init__(self, conv_lstm_kernel_size=3, conv_lstm_out_chan=512, conv_lstm_layer=1):
        super(LSG_NET, self).__init__()
        self.backbone = resnet34()
        self.extractor = nn.Sequential(*list(resnet34().children())[:-2])
        self.resblock = nn.Sequential(*list(resnet34().children())[-3])
        self.lstm = ConvLSTM(1024, conv_lstm_out_chan, (conv_lstm_kernel_size, conv_lstm_kernel_size), conv_lstm_layer,
                             True, True, False)
        self.vo_gap = nn.AdaptiveAvgPool2d((2, 2))
        self.global_gap = nn.AdaptiveAvgPool2d((2, 2))

        self.vo_fc = nn.ModuleList([nn.Linear(2048, 3) for i in range(2)])
        self.global_fc = nn.ModuleList([nn.Linear(2048, 3) for i in range(2)])

    # Input: seq_img, b, s, 3, 224, 224
    # Output: VO_OUT: [(b*(s-1), 3), _], GLOBAL_OUT: [(b*s, 3), _]
    def forward(self, seq_img):
        b = seq_img.size()[0]
        s = seq_img.size()[1]

        features = self.extractor(seq_img.view(b * s, 3, 224, 224)).view(b, s, 512, 7, 7)   # b,s,512,7,7
        concat_features = torch.cat([features[:, :-1], features[:, 1:]], dim=2)             # b,s-1, 1024, 7, 7
        [vo_features], _ = self.lstm(concat_features)                                       # b,s-1, 512, 7, 7

        # soft attention: vo_features and features. For every x_i(global),
        # weighted(channel wise) according to H_1-t(vo) using cosine sim, sum it to x'_i

        # dot product
        dot_prod = torch.sum(torch.sum(features.unsqueeze(2).repeat(1, 1, s - 1, 1, 1, 1)
                                     * vo_features.unsqueeze(1).repeat(1, s, 1, 1, 1, 1), dim=(3, 4, 5)))
        scalar_x = torch.sqrt(torch.sum(torch.square(features.unsqueeze(2).repeat(1, 1, s - 1, 1, 1, 1)), dim=(3,4,5)))
        scalar_h = torch.sqrt(torch.sum(torch.square(vo_features.unsqueeze(1).repeat(1, s, 1, 1, 1, 1)), dim=(3,4,5)))
        weight = dot_prod/(scalar_x*scalar_h)                                               # b, s, s-1

        # x'
        weighted_global_features = torch.sum(weight.view(b,s,s-1, 1,1,1)                    # b, s, 512, 7, 7
                                             * features.unsqueeze(2).repeat(1, 1, s-1, 1, 1, 1), dim=(2,3,4,5))

        vo_feature = self.vo_gap(vo_features.view(b*(s-1), 512, 7, 7)).view(-1, 2048)
        vo_out = [self.vo_fc[i](vo_feature) for i in range(2)]

        global_feature = self.global_gap(weighted_global_features.view(b*s, -1))
        global_out = [self.global_fc[i](global_feature) for i in range(2)]

        return vo_out, global_out


if __name__ == '__main__':
    model = LSG_NET().cuda()
    inp = torch.randn(10, 20, 3, 224, 224).cuda()
    out = model(inp)
    print()
