import torch.nn as nn
import torch
from .register import register_model
from .utils import torch_load, torch_save

cfgs = {
    "resnet18"  : ["3x64-3x64-2", "3x128-3x128-2", "3x256-3x256-2", "3x512-3x512-2"],
    "resnet34"  : ["3x64-3x64-3", "3x128-3x128-4", "3x256-3x256-6", "3x512-3x512-3"],
    "resnet50"  : ["1x64-3x64-1x256-3", "1x128-3x128-1x512-4", "1x256-3x256-1x1024-6", "1x512-3x512-1x2048-3"],
    "resnet101" : ["1x64-3x64-1x256-3", "1x128-3x128-1x512-4", "1x256-3x256-1x1024-23", "1x512-3x512-1x2048-3"],
    "resnet152" : ["1x64-3x64-1x256-3", "1x128-3x128-1x512-8", "1x256-3x256-1x1024-36", "1x512-3x512-1x2048-3"]
}

class Resnet(nn.Module):
    def __init__(self, 
                 cfg=[],
                 num_classes=[],
                 norm_layer=False,
                 drop_rate = 0.):
    
        super().__init__()

        self.cfg = cfg
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.drop_rate = drop_rate

        if self.num_classes <= 0:
            raise ValueError(f"Number classes invalid {self.num_classes} !")
        
        
        if self.norm_layer:
            layers_first_stage = [
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ]
        else:
            layers_first_stage = [
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]

        layers_first_stage += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        self.first_stage = nn.Sequential(*layers_first_stage)

        self.stages = nn.ModuleList()
        in_channels = 64

        for inf in self.cfg:
            inf = inf.split('-')
            num_resi_blocks_per_stage = 0
            inf_layer_per_resi_block = []
            lst_resi_blocks_per_stage = nn.ModuleList()
            for e in inf:
                if 'x' not in e:
                    num_resi_blocks_per_stage = int(e)
                else:
                    ker_size, ker_nums = int(e.split('x')[0]), int(e.split('x')[1])
                    inf_layer_per_resi_block.append([ker_size, ker_nums])
            

            for _ in range(num_resi_blocks_per_stage):
                layers_per_resi_block = []
                for inf_layer in inf_layer_per_resi_block:
                    ker_size, ker_nums = inf_layer[0], inf_layer[1]

                    if self.norm_layer:
                        layers_per_resi_block += [
                            nn.Conv2d(in_channels=in_channels, out_channels=ker_nums, kernel_size=ker_size, stride=1, padding=(ker_size // 2)),
                            nn.BatchNorm2d(ker_nums),
                            nn.ReLU(inplace=True)
                        ]
                    else:
                        layers_per_resi_block += [
                            nn.Conv2d(in_channels=in_channels, out_channels=ker_nums, kernel_size=ker_size, stride=1, padding=(ker_size // 2)),
                            nn.ReLU(inplace=True)
                        ]
                    
                    in_channels = ker_nums
                

                resi_block = nn.Sequential(*layers_per_resi_block)
                lst_resi_blocks_per_stage.append(resi_block)

            self.stages.append(lst_resi_blocks_per_stage)
        
        self.cls = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Dropout(self.drop_rate),
            nn.Linear(3*3*in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.num_classes)
        )
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._initialize_weights()
    
    def forward(self, x):
        x = self.first_stage(x)
        for i, stage in enumerate(self.stages):
            prev = stage[0](x)
            for j, resi_block in enumerate(stage):
                if j > 0:
                    x = resi_block(prev) + prev
                    prev = x
            
            if i != (len(self.stages) - 1):
                x = self.max_pool(x)
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
    

@register_model("resnet18")
def resnet18(num_classes=1000, drop_rate=0.0):
    return Resnet(cfg=cfgs["resnet18"],
                  num_classes=num_classes,
                  drop_rate=drop_rate)

@register_model("resnet18_bn")
def resnet18_bn(num_classes=1000, norm_layer=True, drop_rate=0.0):
    return Resnet(cfg=cfgs["resnet18"],
                  num_classes=num_classes,
                  norm_layer=norm_layer,
                  drop_rate=drop_rate)

@register_model("resnet34")
def resnet34(num_classes=1000, drop_rate=0.0):
    return Resnet(cfg=cfgs["resnet34"],
                  num_classes=num_classes,
                  drop_rate=drop_rate)

@register_model("resnet34_bn")
def resnet34_bn(num_classes=1000, norm_layer=True, drop_rate=0.0):
    return Resnet(cfg=cfgs["resnet34"],
                  num_classes=num_classes,
                  norm_layer=norm_layer,
                  drop_rate=drop_rate)

@register_model("resnet50")
def resnet50(num_classes=1000, drop_rate=0.0):
    return Resnet(cfg=cfgs["resnet50"],
                  num_classes=num_classes,
                  drop_rate=drop_rate)

@register_model("resnet50_bn")
def resnet50_bn(num_classes=1000, norm_layer=True, drop_rate=0.0):
    return Resnet(cfg=cfgs["resnet50"],
                  num_classes=num_classes,
                  norm_layer=norm_layer,
                  drop_rate=drop_rate)

@register_model("resnet101")
def resnet101(num_classes=1000, drop_rate=0.0):
    return Resnet(cfg=cfgs["resnet101"],
                  num_classes=num_classes,
                  drop_rate=drop_rate)

@register_model("resnet101_bn")
def resnet101_bn(num_classes=1000, norm_layer=True, drop_rate=0.0):
    return Resnet(cfg=cfgs["resnet101"],
                  num_classes=num_classes,
                  norm_layer=norm_layer,
                  drop_rate=drop_rate)

@register_model("resnet152")
def resnet152(num_classes=1000, drop_rate=0.0):
    return Resnet(cfg=cfgs["resnet152"],
                  num_classes=num_classes,
                  drop_rate=drop_rate)

@register_model("resnet152_bn")
def resnet152_bn(num_classes=1000, norm_layer=True, drop_rate=0.0):
    return Resnet(cfg=cfgs["resnet152"],
                  num_classes=num_classes,
                  norm_layer=norm_layer,
                  drop_rate=drop_rate)