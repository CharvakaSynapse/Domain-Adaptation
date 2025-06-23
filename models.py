import torch
import torch.nn as nn
import numpy as np

class Classifier(nn.Module):
    def __init__(self, num_classes=31):
        super(Classifier, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return self.fc(x)

class DomainClassifier(nn.Module):
    def __init__(self, num_classes=31, bottleneck=1024, random=True):
        super(DomainClassifier, self).__init__()
        self.random = random
        self.num_classes = num_classes
        self.bottleneck = nn.Linear(2048 * num_classes, bottleneck)
        self.cls_fc = nn.Sequential(
            nn.Linear(bottleneck, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
        if random:
            self.random_matrix = torch.randn(2048 * num_classes, bottleneck) / np.sqrt(2048 * num_classes)
            self.random_matrix = self.random_matrix.cuda()

    def forward(self, feature, softmax_output, grl_lambda):
        feature = feature.view(-1, feature.size(1))
        softmax_output = softmax_output.view(-1, softmax_output.size(1))
        feature_mul = torch.einsum('bi,bj->bij', feature, softmax_output)
        feature_mul = feature_mul.view(feature_mul.size(0), -1)
        if self.random:
            feature_mul = torch.matmul(feature_mul, self.random_matrix)
        feature_mul = GRL(feature_mul, grl_lambda)
        output = self.cls_fc(feature_mul)
        return output.squeeze()

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def GRL(x, lambda_):
    return GradientReversalLayer.apply(x, lambda_)