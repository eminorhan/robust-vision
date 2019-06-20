"""Utility functions
"""
from collections import OrderedDict
import numpy as np
import torch
import torchvision.models
from foolbox.distances import Distance

class Ltwo(Distance):
    """Calculates the mean squared error between two images.
    """
    def _calculate(self):
        min_, max_ = self._bounds
        n = self.reference.size
        f = n * (max_ - min_)**2

        diff = self.other - self.reference
        value = np.linalg.norm(diff) / np.linalg.norm(self.reference)

        # calculate the gradient only when needed
        self._g_diff = diff
        self._g_f = f
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        if self._gradient is None:
            self._gradient = self._g_diff / (self._g_f / 2)
        return self._gradient

    def __str__(self):
        return 'normalized MSE = {:.2e}'.format(self._value)


class CacheModel(torch.nn.Module):
    """define cache model for white-box attacks
    """
    def __init__(self, backbone, mem_keys, mem_vals, theta):
        super(CacheModel, self).__init__()
        self.backbone = backbone
        self.mem_keys = torch.from_numpy(np.float32(mem_keys)).cuda()
        self.mem_vals = torch.from_numpy(mem_vals).cuda()
        self.theta = theta

    def forward(self, x):
        x = self.backbone(x)
        x = torch.squeeze(x)
        x = x / torch.norm(x)

        similarities = torch.exp(self.theta * torch.matmul(x, self.mem_keys))
        p_mem = torch.matmul(similarities, self.mem_vals)
        p_mem = torch.log(p_mem)  # return logits, this makes it easier to run gradient-based attacks
        # p_mem = p_mem / torch.sum(p_mem)

        return p_mem.view(1, 1000)


class LastLayerModel(torch.nn.Module):
    """define a new class specifically for last layer
    """
    def __init__(self, backbone):
        super(LastLayerModel, self).__init__()
        self.backbone = backbone
    def forward(self, x):
        x = self.backbone(x)
        # x = torch.squeeze(x)
        x = torch.nn.functional.softmax(x)
        return x


def load_model(model_name, layer_id, model_save_dir):
    """Loads one of the pretrained SIN models.
    """
    model = torchvision.models.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_save_dir + model_name + '.pth.tar'))

    if layer_id == 176:
        x = list(model.module.children())
    elif layer_id == 175:
        x = list(model.module.children())[:-1]
    elif layer_id == 164:
        x = list(model.module.children())[:-3]
        y = list(model.module.children())[-3]
        z = list(y.children())[:-1]
        x.append(torch.nn.Sequential(*z))
        x.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    if layer_id == 176:
        new_model = LastLayerModel(model)
    else:
        new_model = torch.nn.Sequential(*x)

    return new_model