from __future__ import print_function
import argparse
from sklearn.utils import shuffle
import sys
import argparse
import logging
import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import numpy as np
from torch.utils.data.dataset import Dataset
from dataset import *
from VAEs.VAE_v1 import *
from Discriminator import *
import uproot
import random
#from Segmentation import *



def loss_MSE(recon_x, x, mu, logvar,epoch):
    loss_mse = nn.MSELoss()
    MSE = loss_mse(recon_x, x)
    weight = 2.0
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + weight*KLD, MSE, weight*KLD

def loss_BCE(recon_x, x, mu, logvar,epoch):
    loss_bce = nn.BCELoss()
    weight = 2.0
    BCE = loss_bce(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + weight*KLD, BCE, weight*KLD

# when weight is not equal to 1, then it's beta-VAE,
# and we can improve it by changing weight when training.
def loss_bVAE(recon_x, x, mu, logvar, epoch):
    loss_mse = nn.MSELoss()
    MSE = loss_mse(recon_x,x)
    weight = 1.1
    C = 0.001*epoch
    KLD = torch.abs( -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) - C )
    return MSE+weight*KLD, MSE, weight*KLD


def train2(model, D, device, train_loader, optimizer, optimizer2, epoch, batchsize):
        model.train()
        if epoch%10 == 0:
                D.train()
        for batch_idx, data in enumerate(train_loader):
                data = torch.Tensor(data).to(device).requires_grad_()
                if epoch%10 != 0:
                        optimizer.zero_grad()
                        reconstruction, mu, logvar = model(data)
                        loss, _, _ = loss_MSE(reconstruction, data, mu, logvar, epoch)
                        loss.backward(retain_graph=True)
                        optimizer.step()
                else:
                        optimizer.zero_grad()
                        optimizer2.zero_grad()
                        reconstruction, mu, logvar = model(data)
                        score = D(data)
                        loss1, _, _ = loss_MSE(reconstruction, data, mu, logvar, epoch)
                        _, loss2, _ = loss_BCE(score, torch.ones(batchsize,1).to(device), mu, logvar, epoch)
                        loss = loss1 + 0.2*loss2
                        loss.backward(retain_graph=True)
                        loss2.backward()
                        optimizer.step()
                        optimizer2.step()
               
def test2(model, D, device, test_loader, epoch, batchsize):
        model.eval()
        losses,losses_MSE,losses_BCE=0,0,0
        with torch.no_grad():
                for batch_idx, data in enumerate(test_loader):
                        data = torch.Tensor(data).to(device)
                        reconstruction, mu, logvar = model(data)
                        score = D(data)
                        loss1, bce, kld = loss_MSE(reconstruction, data, mu, logvar, epoch)
                        loss2, bce2, kld2 = loss_BCE(score, torch.ones(batchsize,1).to(device), mu, logvar, epoch)
                        loss = loss1 + 0.2*loss2
                        losses+=loss
                        losses_MSE+=loss1
                        losses_BCE+=loss2
        return losses.cpu().numpy(), losses_MSE.cpu().numpy(), losses_BCE.cpu().numpy()


def get_parser():
        parser = argparse.ArgumentParser(
            description='Run histograms training.', formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument('--datafile', action='store', type=str,
                                                help='ROOT file paths')
        parser.add_argument('--epochs', type=int, default=50, metavar='N',
                                                help='number of epochs to train (default: 50)')
        parser.add_argument('--batchsize', type=int, default=25,
                                                help='size of Batch (default: 25)')
        parser.add_argument('--num_hist', action='store', type=int, default=500,
                                                help='number of histograms that loads')
        parser.add_argument('--lr', type=float, default=1e-7, metavar='LR',
                                                help='learning rate (default: 0.0000001)')
        parser.add_argument('--lr2', type=float, default=1e-5, metavar='LR',
                                                help='learning rate2 (default: 0.00001)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                                help='SGD momentum (default: 0.5)')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                                                help='random seed (default: 1)')
        parser.add_argument('--model_pt_path', action='store', type=str,
                                                help='Model pt path file paths')
        parser.add_argument('--model_restore_pt_path', action='store', type=str,
                                                help='Model restore pt path file paths')
        parser.add_argument('--D_pt_path', action='store', type=str,
                                                help='D Model pt path file paths')
        parser.add_argument('--D_restore_pt_path', action='store', type=str,
                                                help='D Model restore pt path file paths')
        parser.add_argument('--restore', action='store', type=ast.literal_eval, default=False,
                                                help='ckpt file paths')
        return parser


def main():
        logger.info("Start...")
        parser = get_parser()
        parse_args = parser.parse_args()

        datafile = parse_args.datafile
        num_epochs = parse_args.epochs
        learning_rate  = parse_args.lr
        learning_rate2 = parse_args.lr2
        momentum = parse_args.momentum
        seed = parse_args.seed
        model_pt_path = parse_args.model_pt_path
        model_restore_pt_path = parse_args.model_restore_pt_path
        D_pt_path = parse_args.D_pt_path
        D_restore_pt_path = parse_args.D_restore_pt_path
        num_hist = parse_args.num_hist
        batchsize = parse_args.batchsize
        restore = parse_args.restore

        # --- set up all the logging stuff
        formatter = logging.Formatter(
             '%(asctime)s - %(name)s'
             '[%(levelname)s]: %(message)s'
        )
        hander = logging.StreamHandler(sys.stdout)
        hander.setFormatter(formatter)
        logger.addHandler(hander)
        #########################################
        logger.info('constructing graph')

        #Set Random Seed
        torch.manual_seed(seed)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(device)

        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

        #load the data.
        logger.info('Start loading data:')
        logger.info(datafile)
        Dataset_Train = HistoDataset(datafile,num_hist)
        train_loader = torch.utils.data.DataLoader(Dataset_Train, batch_size=batchsize,
                                                shuffle=True, num_workers=0)
        Dataset_Test = HistoDataset(datafile,num_hist)
        test_loader = torch.utils.data.DataLoader(Dataset_Test, batch_size=batchsize,
                                                shuffle=False, num_workers=0)
        logger.info('Load data successfully! Start training...')

        # first try: only train VAE
        model = CVAE().to(device)
        D = DNet().to(device)
        if restore:
                model.load_state_dict(torch.load(model_restore_pt_path))
                D.load_state_dict(torch.load(D_restore_pt_path))
                logger.info('Load model from:')
                logger.info(model_restore_pt_path)
                logger.info(D_restore_pt_path)
        #UNet = AttU_Net3D().to(device)

        
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        optimizer  = optim.Adam(model.parameters(), lr=learning_rate,  betas=(0.5, 0.999))
        optimizer2 = optim.Adam(D.parameters(), lr=learning_rate2, betas=(0.5, 0.999))
        #optimizer2 = optim.SGD(UNet.parameters(), lr=args.lr, momentum=args.momentum)
        
        for epoch in range( 1,num_epochs+1 ):
                train2(model, D, device, train_loader, optimizer, optimizer2, epoch, batchsize)
                #train(model, device, train_loader, optimizer, epoch, batchsize)
                if epoch == 1:
                        logger.info("1 epoch completed! This code is running successfully!")
                if epoch%(num_epochs//40)==0:
                        losses,losses_MSE,losses_BCE = test2(model, D, device, test_loader, epoch, batchsize)
                        logger.info( "Epoch %6d. Loss %5.3f. Loss_MSE %5.3f. Loss_BCE %5.3f." % ( epoch, losses, losses_MSE, losses_BCE ) )
                        #losses = test(model, device, test_loader, epoch, batchsize)
                        #logger.info( "Epoch %6d. Loss %5.3f." % ( epoch, losses ) )
                if epoch%100==0:
                        torch.save(model.state_dict(), model_pt_path)
                        torch.save(D.state_dict(), D_pt_path)
                        logger.info('Model save into:')
                        logger.info(model_pt_path)
                        logger.info(D_pt_path)

        logger.info("Train Done!")

if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)

if __name__ == '__main__':
    main()
