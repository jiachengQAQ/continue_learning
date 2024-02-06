import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from convs.conv_cifar import conv3
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
import copy
import time
from dd_algorithms.utils import DiffAugment,ParamDiffAug,get_time, match_loss, TensorDataset, epoch
import pdb
# todo
Iteration = 1000
# Iteration = 1
ipc = 10
lr_img = 0.1
lr_net = 0.01
dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
BN  =True
channel = 3
dsa = False if dsa_strategy in ['none', 'None'] else True
im_size= [32,32]
batch_real = 256
device = 'cuda:3'
dis_metric = 'ours'
outer_loop, inner_loop = 10, 50
# num_classes = 100
init = True
class GradientMatching():
    def __init__(self, args,pretrained = False):

        self._device = args["device"][0]
        self.dsa_param = ParamDiffAug()
        self.pretrained = pretrained
        if self.pretrained:
            fname = 'cl_res_DSA_CIFAR100_ConvNet_20ipc_10steps_seed0.pt'
            data = torch.load(fname, map_location='cpu')['data']
            self.images_train_all = data[0][0]
            self.labels_train_all = data[0][1]

    def gen_synthetic_data(self, old_model, initial_data, real_data, real_label, class_range, image_syn=None):
        if self.pretrained:
            step = (class_range[-1]+1)//10
            images_train = torch.tensor(self.images_train_all[(step-1)*ipc*len(class_range):step*ipc*len(class_range)])
            labels_train = torch.tensor(self.labels_train_all[(step-1)*ipc*len(class_range):step*ipc*len(class_range)])
            print(labels_train)
            new_syn = [images_train,labels_train]
            return new_syn
        images_all = real_data
        labels_all = real_label
        indices_class = {c:[] for c in class_range}
        #indices_class = [[] for c in range (class_range)]

        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.tensor(images_all,dtype=torch.float).to(self._device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self._device)

        # ''' organize the real dataset '''
        # images_all = []
        # labels_all = []
        # indices_class = [[] for c in range(num_classes)]
        #
        # images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        # labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        # for i, lab in enumerate(labels_all):
        #     indices_class[lab].append(i)
        # images_all = torch.cat(images_all, dim=0).to(args.device)
        # labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)


        for c in class_range:
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            replace = False if len(indices_class[c]) >= n else True
            idx_shuffle = np.random.choice(indices_class[c], size=n, replace=replace)
            return images_all[idx_shuffle]
        # for ch in range(channel):
        #     print('real images channel %d, mean = %.4f, std = %.4f' % (
        #         ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch]))
        #
        # image_syn = torch.randn(size=(len(class_range)*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=self._device)
        # label_syn = torch.tensor(np.array([np.ones(ipc)*i for i in class_range]), dtype=torch.long, requires_grad=False, device=self._device).view(-1)
        # optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_img, momentum=0.5) # optimizer_img for synthetic data
        # optimizer_img.zero_grad()
        # criterion = nn.CrossEntropyLoss().to(device)
        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f' % (
            ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(len(class_range) * ipc, channel, im_size[0], im_size[1]), dtype=torch.float,
                                requires_grad=True, device=self._device)
        label_syn = torch.tensor([np.ones(ipc) * i for i in class_range], dtype=torch.long,
                                 requires_grad=False, device=self._device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]

        if init == 'real':
            print('initialize synthetic data from random real images')
            for c in class_range:
                image_syn.data[c * ipc:(c + 1) * ipc] = get_images(c, ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_img, momentum=0.5)  # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(self._device)
        print('%s training begins' % get_time())

        for it in tqdm(range(Iteration+1)):

            ''' Train synthetic data '''
            # net = conv3(None,None) # get a random model
            net = old_model.copy().activate()
            net = net.to(self._device)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()

            # for param in list(net.parameters()):
            #     param.requires_grad = False

            loss_avg = 0
            for ol in range(outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  # BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in class_range], dim=0)
                    net.train()  # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real)  # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  # BatchNorm
                            module.eval()  # fix mu and sigma of every BatchNorm layer

                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(self._device)
                for c in class_range:
                    related_class = c - class_range[0]
                    img_real = get_images(c, batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=self._device, dtype=torch.long) * c
                    #img_syn1 = image_syn[c * ipc:(c + 1) * ipc].reshape(
                    #    (ipc, channel, im_size[0], im_size[1]))
                    img_syn = image_syn[related_class * ipc:(related_class + 1) * ipc].reshape(
                        (ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((ipc,), device=self._device, dtype=torch.long) * c

                    if dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, dsa_strategy, seed=seed, param=self.dsa_param)
                        img_syn = DiffAugment(img_syn, dsa_strategy, seed=seed, param=self.dsa_param)

                    output_real = net(img_real)['logits']
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)['logits']
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, dis_metric,self._device)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == outer_loop - 1:
                    break

                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                    label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=batch_real, shuffle=True,
                                                          num_workers=0)
                for il in range(inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, device, self.dsa_param,aug=True)

            loss_avg /= (len(class_range) * outer_loop)

            if it % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == Iteration:  # only record the final results
                new_syn = [copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())]
                logging.info("Exemplar size: {}".format(len(new_syn[0])))
                return new_syn













        #     "previous"
        #     ''' update synthetic data '''
        #     if not BN: # for ConvNet
        #         loss = torch.tensor(0.0).to(self._device)
        #         for c in range(class_range):
        #             related_class = c-class_range[0]
        #             img_real = get_images(c, batch_real)
        #             img_syn = image_syn[related_class*ipc:(related_class+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
        #
        #             if dsa:
        #                 seed = int(time.time() * 1000) % 100000
        #                 img_real = DiffAugment(img_real, dsa_strategy, seed=seed, param=self.dsa_param)
        #                 img_syn = DiffAugment(img_syn, dsa_strategy, seed=seed, param=self.dsa_param)
        #
        #             output_real = net(img_real).detach()
        #             output_syn = net(img_syn)
        #
        #             loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
        #
        #     else: # for BN
        #         images_real_all = []
        #         images_syn_all = []
        #         labels_real_all = []
        #         labels_syn_all = []
        #         loss = torch.tensor(0.0).to(self._device)
        #         for c in class_range:
        #             related_class = c-class_range[0]
        #             img_real = get_images(c, batch_real)
        #             lab_real = torch.ones((img_real.shape[0],), device=device, dtype=torch.long) * c
        #             img_syn = image_syn[related_class*ipc:(related_class+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
        #             # import pdb
        #             # pdb.set_trace()
        #             #img_syn = image_syn[c * ipc:(c + 1) * ipc].reshape(
        #             #    (ipc, channel, im_size[0], im_size[1]))
        #             lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c
        #
        #             if dsa:
        #                 seed = int(time.time() * 1000) % 100000
        #                 img_real = DiffAugment(img_real, dsa_strategy, seed=seed, param=self.dsa_param)
        #                 img_syn = DiffAugment(img_syn, dsa_strategy, seed=seed, param=self.dsa_param)
        #
        #             images_real_all.append(img_real)
        #             images_syn_all.append(img_syn)
        #             labels_real_all.append(lab_real)
        #             labels_syn_all.append(lab_syn)
        #
        #         images_real_all = torch.cat(images_real_all, dim=0)
        #         images_syn_all = torch.cat(images_syn_all, dim=0)
        #         labels_real_all = torch.cat(labels_real_all, dim=0)
        #         labels_syn_all = torch.cat(labels_syn_all, dim=0)
        #
        #         output_real = net(images_real_all)['logits']
        #         loss_real = criterion(output_real, labels_real_all)
        #         gw_real = torch.autograd.grad(loss_real, net_parameters)
        #         gw_real = list((_.detach().clone() for _ in gw_real))
        #
        #
        #         output_syn = net(images_syn_all)['logits']
        #         loss_syn = criterion(output_syn, labels_syn_all)
        #         gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
        #
        #         loss += match_loss(gw_syn, gw_real, dis_metric, device)
        #
        #     optimizer_img.zero_grad()
        #     loss.backward()
        #     optimizer_img.step()
        #     loss_avg += loss.item()
        #     loss_avg /= len(class_range)
        #
        #     if it%1000 == 0:
        #         print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))
        #
        #
        # new_syn = [copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())]
        #
        #
        # logging.info("Exemplar size: {}".format(len(new_syn[0])))
        # # pdb.set_trace()
        # return new_syn
