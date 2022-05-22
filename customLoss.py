import torch

class FalsePositiveManglerLoss:

    def __init__(self, device):
        loss_weights = torch.tensor([10, 1]).to(device).float()
        self.ce = torch.nn.CrossEntropyLoss(loss_weights)
        self.device = device


    def __call__(self, x, y):
        class1 = torch.ones(x.shape).to(self.device) - x  # background prediciton - we want to punish this
        class2 = x #foreground prediction

        input = torch.cat([class1, class2], axis=1)

        y += 0.1
        y = y.long().squeeze(1)

        return self.ce(input, y)

