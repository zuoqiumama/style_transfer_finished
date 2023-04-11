import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import models


def gram_matrix(tensor):
    _, depth, height, width = tensor.size()
    tensor = tensor.view(depth, height * width)
    gram = torch.mm(tensor, tensor.t())
    return gram


def get_features(image, net, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in net._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


class Transfer:
    def __init__(self, device, content, style, style_weights, content_weight, style_weight):
        self.device = device
        self.net = models.vgg19(pretrained=True).features
        for param in self.net.parameters():
            param.requires_grad_(False)
        self.net.to(device)
        self.content = content
        self.style = style
        # 每层提取出来的特征的损失权重
        self.style_weights = style_weights
        # 风格损失总体权重
        self.style_weight = style_weight
        self.content_weight = content_weight

    def train(self, epochs, lr):
        content_features = get_features(self.content, self.net)
        style_features = get_features(self.style, self.net)
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
        target = self.content.clone().requires_grad_(True).to(self.device)
        optimizer = optim.Adam([target], lr)
        for ii in range(1, epochs + 1):
            # 获取目标图像特征
            target_features = get_features(target, self.net)
            # 计算内容损失
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
            # 计算风格损失
            style_loss = 0
            # 计算风格层每层GRAM矩阵与目标层GRAM矩阵的损失
            for layer in self.style_weights:
                # 获取目标图像的风格层并将其转为GRAM矩阵
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                _, depth, height, width = target_feature.shape
                # 获取当前层的风格层GRAM矩阵
                style_gram = style_grams[layer]
                layer_style_loss = self.style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                # add to the style loss
                style_loss += layer_style_loss / (depth * height * width)
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            # 训练
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return target
