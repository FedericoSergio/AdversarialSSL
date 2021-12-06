import torch as ch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import attacker

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False), nn.ReLU(), self.backbone.fc)

        self.attacker = attacker.Attacker(self.backbone)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, inp, target, make_adv, **attacker_kwargs):
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            adv = self.attacker(inp, target, **attacker_kwargs)
            if prev_training:
                self.train()

            inp = adv

        return (self.backbone(inp), inp)