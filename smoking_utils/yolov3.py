import torch.nn as nn

# DETECT_CLASSES = (
#     "smoke",
#     "phone", 
#     "antistatic_cap"
# )

DETECT_CLASSES = (
    "phone",
    "smoke", 
    "cup",
    "cap",
    "helmet"
)

class Exp(object):
    def __init__(self):
        self.num_classes = len(DETECT_CLASSES)
        self.width = 1.0

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from .models import YOLOFPN, YOLOFPN_DMS, YOLOXHead, YOLOX
            backbone = YOLOFPN_DMS()
            head = YOLOXHead(self.num_classes, self.width, in_channels=[128, 128, 128], act="relu")
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model