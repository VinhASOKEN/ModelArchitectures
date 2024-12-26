import torch.nn as nn
from .register import register_model
from .utils import torch_load, torch_save

cfgs = {
    "vgg11" : [64, "MP", 128, "MP", 256, 256, "MP", 512, 512, "MP", 512, 512, "MP"],
    "vgg13" : [64, 64, "MP", 128, 128, "MP", 256, 256, "MP", 512, 512, "MP", 512, 512, "MP"],
    "vgg16" : [64, 64, "MP", 128, 128, "MP", 256, 256, 256, "MP", 512, 512, 512, "MP", 512, 512, 512, "MP"],
    "vgg19" : [64, 64, "MP", 128, 128, "MP", 256, 256, 256, 256, "MP", 512, 512, 512, 512, "MP", 512, 512, 512, 512, "MP"]
}

class VGG(nn.Module):
    def __init__(self,
                 cfg = [],
                 num_classes = 1000,
                 norm_layer = False,
                 drop_rate = 0.):
        super().__init__()

        self.cfg = cfg
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.drop_rate = drop_rate

        if self.num_classes <= 0:
            raise ValueError(f"Number classes invalid {self.num_classes} !")

        in_channels = 3
        layers = []

        for inf in self.cfg:
            if inf == "MP":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = int(inf)
                if self.norm_layer:
                    layers += [
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ]
                
                in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.drop_rate),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes)
        )

        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = self.cls(x)

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def save(self, filepath):
        return torch_save(self, filepath)

    @classmethod
    def load(cls, filepath):
        print(f'Loading weight from {filepath}')
        return torch_load(filepath)


@register_model("vgg11")
def vgg11(num_classes = 1000, drop_rate=0.0):
    return VGG(cfg=cfgs["vgg11"],
               num_classes=num_classes,
               drop_rate=drop_rate)

@register_model("vgg11_bn")
def vgg11_bn(num_classes = 1000, norm_layer=True, drop_rate=0.0):
    return VGG(cfg=cfgs["vgg11"],
               num_classes=num_classes,
               norm_layer=norm_layer,
               drop_rate=drop_rate)

@register_model("vgg13")
def vgg13(num_classes = 1000, drop_rate=0.0):
    return VGG(cfg=cfgs["vgg13"],
               num_classes=num_classes,
               drop_rate=drop_rate)

@register_model("vgg13_bn")
def vgg13_bn(num_classes = 1000, norm_layer=True, drop_rate=0.0):
    return VGG(cfg=cfgs["vgg13"],
               num_classes=num_classes,
               norm_layer=norm_layer,
               drop_rate=drop_rate)

@register_model("vgg16")
def vgg16(num_classes = 1000, drop_rate=0.0):
    return VGG(cfg=cfgs["vgg16"],
               num_classes=num_classes,
               drop_rate=drop_rate)

@register_model("vgg16_bn")
def vgg16_bn(num_classes = 1000, norm_layer=True, drop_rate=0.0):
    return VGG(cfg=cfgs["vgg16"],
               num_classes=num_classes,
               norm_layer=norm_layer,
               drop_rate=drop_rate)

@register_model("vgg19")
def vgg19(num_classes = 1000, drop_rate=0.0):
    return VGG(cfg=cfgs["vgg19"],
               num_classes=num_classes,
               drop_rate=drop_rate)

@register_model("vgg19_bn")
def vgg19_bn(num_classes = 1000, norm_layer=True, drop_rate=0.0):
    return VGG(cfg=cfgs["vgg19"],
               num_classes=num_classes,
               norm_layer=norm_layer,
               drop_rate=drop_rate)