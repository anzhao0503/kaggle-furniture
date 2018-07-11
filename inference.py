import os, sys
import numpy as np
from PIL import Image
import math
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import densenet
from dataloader import *


def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k / sum_y for k in y]
    return z


def main():
    model_path = 'densenet2_epoch8.pth'
    data_dir = '../data/datasets/'
    num_classes = 128
    new_height = 400
    new_width = 400
    batch_size = 10
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
                        transforms.TenCrop(384),
                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns 4D
                        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=mean, std=std)(crop) for crop in crops]))  # returns 4D
                        # transforms.ToTensor(),
                        # transforms.Normalize(mean=mean, std=std)
                        ])
    test_data = MyDataset(os.path.join(data_dir, 'test1.txt'), data_dir, new_width, new_height, test_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    net = torch.load(model_path)
    # net = densenet.densenet161(pretrained=True, num_classes=128)
    # net = torch.nn.DataParallel(net).cuda()
    net = net.cuda()
    # net.load_state_dict(params['state_dict'])

    net.eval()
    print "Loading model ok..."
    lists = []
    pres = np.zeros((12800, 1))
    # pros = np.zeros((12800, 128))
    # pros1 = np.zeros((12800, 19))
    for i, (input, _, idx) in enumerate(test_loader):
        input = input.float().cuda(async=True)
        bs, ncrops, c, h, w = input.size()
        input_var = torch.autograd.Variable(input.view(-1, c, h, w), volatile=True)
        # compute output
        output = net(input_var)
        # print output
        predict = output.data.view(bs, ncrops, -1).mean(1)
        # predict1 = output1.data
        _, pred = predict.topk(1, 1, True, True)
        pred = pred.t()
        pred = pred.view(-1, 1)
        pred = pred.cpu().numpy()
        
        # predict = np.argmax(np.bincount(pred[0]))
        
        for j in range(batch_size):
            # print idx[j]
            pres[int(idx[j]) - 1, :] = pred[j][0]
            print pred[j][0]
            # pros[int(idx[j]) - 1, :] = predict[j]
            # pros1[int(idx[j]) - 1, :] = predict1[j]
            lists.append(int(idx[j])-1)
    # np.savetxt("resnet152_mulloss_genuspre.txt", pros1)
    
    csv_file = open("result_densenet161_tta_new.csv", "w")
    writer = csv.writer(csv_file)
    file_header = ["id", "predicted"]
    writer.writerow(file_header)
    for i in range(12800):
        if i not in lists:
            predict_label = 19
        else:
            predict_label = pres[i] + 1
        row = [str(i + 1), str(int(predict_label))]
        writer.writerow(row)
    csv_file.close()


if __name__ == '__main__':
    main()
