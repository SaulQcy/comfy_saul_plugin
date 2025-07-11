import torch
import torch.nn as nn

from .darknet import Darknet
from .network_blocks import BaseConv


class Conv(nn.Module):
      def __init__(self, in_channels, out_channels, kernel_size, stride=1):
          super(Conv, self).__init__()
          self.conv = nn.Sequential(
              nn.Conv2d(
                  in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  kernel_size // 2,
                  bias=False,
              ),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True),
          )

      def forward(self, x):
         return self.conv(x)


class Upsample(nn.Module):
      def __init__(self, in_channels, out_channels, scale=2):
          super(Upsample, self).__init__()
          self.upsample = nn.Sequential(
              Conv(in_channels, out_channels, 1),
              nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1, bias=False),
              nn.BatchNorm2d(out_channels)
          )

      def forward(self, x):
          return self.upsample(x)


class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 21 is the default backbone of this model.
    """
    def __init__(
        self,
        depth=21,
        in_features=["dark3", "dark4", "dark5"]
    ):
        super().__init__()
        self.backbone = Darknet(depth)
        self.in_features = in_features

        # out 1
        self.out1 = self._make_embedding([128, 128], 128)

        # out 2
        self.out2 = self._make_embedding([128, 128], 128)

        # self.out3 = self._make_embedding2([128, 128], 128)

        # upsample
        #self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.Upsample1 = Upsample(128, 128)
        self.Upsample2 = Upsample(128, 128)
        self.Relu = nn.ReLU(inplace=True)

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="relu")
    
    def _make_cbl2(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=2, act="relu")

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                #self._make_cbl(filters_list[1], filters_list[0], 1),
                #self._make_cbl(filters_list[0], filters_list[1], 3),
                #self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m

    def _make_embedding2(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl2(filters_list[0], filters_list[1], 3)
            ]
        )
        return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print("loading pretrained weights...")
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        #  backbone
        out_features = self.backbone(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]

        #  yolo branch 1
        x1_in = self.Upsample1(x0)
        x1_in = x1_in + x1
        x1_in = self.Relu(x1_in)

        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.Upsample2(out_dark4)
        x2_in = x2_in + x2
        x2_in = self.Relu(x2_in)

        out_dark3 = self.out2(x2_in)

        outputs = (out_dark3, out_dark4, x0)
        return outputs
