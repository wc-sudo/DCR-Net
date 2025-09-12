import os
import argparse
from glob import glob
import numpy as np
from model import DCRetNet

parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_id', dest='gpu_id', default="0",
                    help='GPU ID (-1 for CPU)')
parser.add_argument('--epochs', dest='epochs', type=int, default=200,
                    help='number of total epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16,
                    help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=96,
                    help='patch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--data_dir', dest='data_dir',
                    default=r'/home/pilab/demo/KinD_plus-master/LOLdataset',
                    help='directory storing the training data')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./ckpts/',
                    help='directory for checkpoints')

args = parser.parse_args()


def train(model):

    lr = args.lr * np.ones([args.epochs])
    lr[10:] = lr[0] / 10.0

    train_low_data_names = glob(args.data_dir + '/our485/low/*.png')
                           #  + \
                           # glob(args.data_dir + '/low/*.jpg'))

    train_low_data_names.sort()
    train_high_data_names = glob(args.data_dir + '/our485/high/*.png')
                            # + \
                            # glob(args.data_dir + '/Nikon/high/*.jpg')

    train_high_data_names.sort()
    eval_low_data_names = glob('C:/Users/dell/Desktop/data/eval/low/*.png')
    eval_low_data_names.sort()
    eval_high_data_names = glob('C:/Users/dell/Desktop/data/eval/high/*.png')
    eval_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('Number of training data: %d' % len(train_low_data_names))

    # 配置参数保持不变
    ID_config = {
        'batch_size': 10,
        'patch_size': 48,
        'epoch': 200,
        'lr': np.ones(2400) * 0.0001
    }

    RR_lr = np.zeros(2400)
    for i in range(2400):
        if i <= 800:
            RR_lr[i] = 0.0001
        elif 801 <= i <= 1250:
            RR_lr[i] = 0.0001 * 0.5
        elif 1251 <= i <= 1500:
            RR_lr[i] = 0.0001 * 0.25
        else:
            RR_lr[i] = 0.0001 * 0.1

    RR_config = {
        'batch_size': 4,
        'patch_size': 128,
        'epoch': 2400,
        'lr': RR_lr
    }

    IC_config = {
        'batch_size': 10,
        'patch_size': 48,
        'epoch': 2400,
        'lr': np.ones(2400) * 0.0001
    }
    model.train(train_low_data_names,
                train_high_data_names,
                eval_low_data_names,
                eval_high_data_names,
                batch_size=ID_config['batch_size'],
                patch_size=ID_config['patch_size'],
                epoch=ID_config['epoch'],
                lr=ID_config['lr'],
                vis_dir=args.vis_dir,
                ckpt_dir=args.ckpt_dir,
                eval_every_epoch=100,
                train_phase="Decom")

    model.train(train_low_data_names,
                train_high_data_names,
                eval_low_data_names,
                eval_high_data_names,
                batch_size=RR_config['batch_size'],
                patch_size=RR_config['patch_size'],
                epoch=RR_config['epoch'],
                lr=RR_config['lr'],
                vis_dir=args.vis_dir,
                ckpt_dir=args.ckpt_dir,
                eval_every_epoch=100,
                train_phase="Denoise")

    model.train(train_low_data_names,
                train_high_data_names,
                eval_low_data_names,
                eval_high_data_names,
                batch_size=IC_config['batch_size'],
                patch_size=IC_config['patch_size'],
                epoch=IC_config['epoch'],
                lr=IC_config['lr'],
                vis_dir=args.vis_dir,
                ckpt_dir=args.ckpt_dir,
                eval_every_epoch=100,
                train_phase="Relight")


if __name__ == '__main__':
    if args.gpu_id != "-1":
        # Create directories for saving the checkpoints and visuals
        args.vis_dir = args.ckpt_dir + '/visuals/'
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.vis_dir):
            os.makedirs(args.vis_dir)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        model = DCRetNet().cuda()
        train(model)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
