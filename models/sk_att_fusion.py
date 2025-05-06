import torch
from torch import nn

class SK_channel_my(nn.Module):
    def __init__(self, features, features_o, stride=1):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SK_channel_my, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv1d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Conv1d(features, features, kernel_size=5, stride=1, padding=1, bias=False)
        self.act2 = nn.Sigmoid()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(features,features_o,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(features_o),
            nn.SiLU()
        )

    def forward(self, x):
        feats_U = x
        feats_S = self.gap(feats_U)
        # B,C,1,1--B,C,1--conv1d--B,C,1,1
        feats_Z = self.fc1(feats_S.squeeze(-1)).unsqueeze(-1) #torch.Size([2, 32, 1, 1])
        feats_Z = self.act1(feats_Z) # B,M,C,1,1
        feats_Z = self.fc1(feats_Z.squeeze(-1)).unsqueeze(-1) #torch.Size([2, 32, 1, 1])
        feats_Z = self.act2(feats_Z) # B,M,C,1,1
        feats_V = feats_U * feats_Z   # (B,C,H,W)*(B,C,1,1)--# B,C,H,W
        feats_V = self.fusion_conv(feats_V)

        return feats_V