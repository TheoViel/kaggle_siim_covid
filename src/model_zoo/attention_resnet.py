import math
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter


class ConvAttentionResNet(nn.Module):
    """
    ResNet with a convolutional attention mechanism.
    """

    def __init__(self, resnet, reduce_stride_3=False):
        """
        Constructor.

        Args:
            resnet (ResNet): ResNet to build the model from.
            reduce_stride_3 (bool, optional): Change the layer3 stride to (1, 1). Defaults to False.
        """
        super().__init__()
        self.resnet = resnet

        self.num_classes = resnet.num_classes
        self.nb_ft = resnet.fc.in_features

        self.att_conv = nn.Sequential(
            nn.Conv2d(self.nb_ft, self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(
                self.num_classes, self.num_classes, kernel_size=3, padding=1, bias=False
            ),
        )
        self.value_conv = nn.Conv2d(self.nb_ft, self.nb_ft, kernel_size=3, padding=1)

        std = 1. / math.sqrt(self.nb_ft)
        self.logits_w = Parameter(torch.Tensor(1, self.nb_ft, self.num_classes).uniform_(-std, std))
        self.logits_bias = Parameter(torch.Tensor(1, self.num_classes).fill_(0))

        if "resnext" in resnet.name:
            self.resnet.layer4[0].conv2.stride = (1, 2)
        else:
            self.resnet.layer4[0].conv1.stride = (1, 2)
        self.resnet.layer4[0].downsample[0].stride = (1, 2)

        if reduce_stride_3:
            self.resnet.layer3[0].conv1.stride = (1, 1)
            self.resnet.layer3[0].downsample[0].stride = (1, 1)

        # self.resnet.conv1.stride = (1, 1)

    def attention_mechanism(self, fts):
        """
        Performs the forward pass of the attention mechanism.

        Args:
            fts (torch tensor [batch_size x nb_ft x h x w]): Feature maps.

        Returns:
            torch tensor (batch_size x num_classes x nb_ft): Pooled features.
            torch tensor (batch_size x h x w): Attention maps.
        """
        bs, c, h, w = fts.size()

        att = self.att_conv(fts)

        att = torch.softmax(att.view(bs, -1, h * w), -1).transpose(1, 2)

        fts = self.value_conv(fts)
        fts = fts.view(bs, c, h * w)

        return torch.bmm(fts, att), att.view(bs, h, w, -1)

    def forward(self, x, return_att=False):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x 3 x w x h]): Image
            return_att (bool, optional): Whether to return the attention maps. Defaults to False.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes x w x h]: Attention maps, if return_att is True.
            0: placeholder since auxiliary logits are not computed.
        """
        fts = self.resnet.extract_fts(x)

        pooled, att = self.attention_mechanism(fts)

        logits = (pooled * self.logits_w).sum(1) + self.logits_bias

        if return_att:
            return att, logits, 0

        return logits, 0


class ConvAttentionResNetAux(ConvAttentionResNet):
    def __init__(self, resnet, reduce_stride_3=False, num_classes_aux=6):
        """
        Constructor

        Args:
            resnet (torch ResNet): ResNet backbone.
            reduce_stride_3 (bool, optional): Change the layer3 stride to (1, 1). Defaults to False.
            num_classes_aux (int, optional): Number of auxiliary classes. Defaults to 6.
        """
        super().__init__(resnet, reduce_stride_3=reduce_stride_3)

        self.num_classes_aux = num_classes_aux
        self.logits_aux = nn.Linear(self.nb_ft, num_classes_aux)

    def forward(self, x, return_att=False, return_ft=False):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x 3 x w x h]): Image
            return_att (bool, optional): Whether to return the attention maps. Defaults to False.
            return_ft (bool, optional): Whether to return the pooled features. Defaults to False.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: Auxiliary logits.
            torch tensor [batch_size x num_classes x w x h]: Attention maps, if return_att is True.
            torch tensor [batch_size x num_classes x w x h]: Pooled features, if return_ft is True.
        """
        fts = self.resnet.extract_fts(x)

        pooled, att = self.attention_mechanism(fts)

        logits = (pooled * self.logits_w).sum(1) + self.logits_bias

        logits_aux = self.logits_aux(pooled.mean(-1))

        if return_att:
            return att, logits, logits_aux
        elif return_ft:
            return pooled, logits, logits_aux

        return logits, logits_aux
