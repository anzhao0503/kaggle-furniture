import torch
import torch.nn


class PairwiseConfusion(torch.nn.Module):
    def __init__(self):
        super(PairwiseConfusion, self).__init__()

    def forward(self, batch_left, batch_right, label):
        if batch_left.size(0) != batch_right.size(0):
            raise Exception('Incorrect batch size provided')
        batch_left_new = batch_left
        batch_right_new = batch_right
        sum = 0
        for i in range(batch_left.size(0)):
            sum += torch.norm((batch_left_new[i] - batch_right_new[i]).abs() * label[i])
        loss = sum / float(batch_left.size(0) * 2)
        return loss
