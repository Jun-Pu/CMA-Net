import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.CMANet import CMANet
from data import test_dataset_2
import time


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default=os.getcwd() + '/LFSOD_dataset/RGBD_for_test/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = CMANet()
model.load_state_dict(torch.load(os.getcwd() + '/model_pths/CMANet_final.pth'))
model.cuda()
model.eval()

#test
test_datasets = ['DUT-LF', 'HFUT']
TIME = []
for dataset in test_datasets:
    save_path = './Predictions/CMANet/ResNet50/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset_2(image_root, gt_root, depth_root, opt.testsize, dataset)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data(dataset)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        time_s = time.time()
        res, _, _ = model(image, depth)
        time_e = time.time()
        TIME.append(time_e - time_s)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ', save_path+name)
        cv2.imwrite(save_path+name, res*255)
    print('Test Done!')
print('Speed: %f FPS' % (617 / np.sum(TIME)))
print('Test Done!')
