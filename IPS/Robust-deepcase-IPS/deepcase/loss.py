from   torch.autograd import Variable
import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):

    def __init__(self, size, smoothing=0.0):
        """Implements label smoothing loss function

            Parameters
            ----------
            size : int
                Number of labels-- 100

            smoothing : float, default=0.0 --0.1
                Smoothing factor to apply
            """
        # Initialise super
        super(LabelSmoothing, self).__init__()
        # Set KL-Divergence loss function
        self.criterion      = nn.KLDivLoss(reduction='none')
        self.criterion_attn = nn.MSELoss()
        # Set size
        self.size = size
        # Set confidence and smoothing
        self.smoothing  =       smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target, weights=None, attention=None):
        """Forward data"""
        # Check if shape of data is the same as expected
        assert x.size(-1) == self.size
        # Set target if necessary
        if x.ndim != target.ndim:
            target = target.unsqueeze(-1)

        # Create true distribution
        true_dist = x.data.clone()
        # print('true_dist0:',true_dist)
        true_dist.fill_(self.smoothing / (self.size - 2))
        # print('true_dist.fill:', true_dist)
        true_dist.scatter_(1, target, self.confidence)
        # print('true_dist.scatter:', true_dist)

        # Apply criterion
        y_target = Variable(true_dist, requires_grad=False)
        # print('x.shape:', x.shape)
        # print('y_target.shape:', y_target.shape)
        result = self.criterion(x, y_target)

        # Apply weights if necessary
        if weights is not None:
            result *= weights.to(torch.float).unsqueeze(1)
        # Get result as sum
        result = result.sum()

        # Apply attention criterion if necessary
        if attention is not None:
            target = torch.ones(attention.shape, device=x.device) / attention.shape[1]
            return result + 0.05*self.criterion_attn(attention, target)
        else:
            return result
