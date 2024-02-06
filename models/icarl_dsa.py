import logging
import pdb

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
from PIL import Image
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy,denormalize_cifar100,tensor2img
from dd_algorithms.dsa import GradientMatching
import time
from dd_algorithms.utils import DiffAugment,ParamDiffAug,get_time
EPSILON = 1e-8
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
#init_epoch = 200
init_epoch = 1
init_lr = 0.01
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005
step_syn = 3

epochs = 170
#epochs = 1
lrate = 0.01
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
weight_decay = 1e-5
num_workers = 8
T = 2
dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
use_pretrained = True

class iCaRL_DSA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.dd = GradientMatching(args)
        self.dsa_param = ParamDiffAug()
        self.dsa_strategy = dsa_strategy

    def after_task(self):
        self._known_classes = self._total_classes
        
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.syn_loader = None
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        if self._get_memory() is not None:
            train_dataset,syn_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="train",
                appendent=self._get_memory(),
                is_dd = True
            )
            self.syn_loader = DataLoader(
                syn_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
        else:
            train_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train",
                mode="train",
                appendent=self._get_memory(),
                is_dd = True
            )


        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        #xxxxxxxxxxxxxxxxxxxxxxxx
        

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader,self.syn_loader, self.test_loader)
        self._old_network = self._network.copy().freeze()
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader,syn_loader, test_loader,use_pretrained=False):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            if use_pretrained:
                f = open('./ini_resnet18_cifar100', 'rb')
                self._network = pickle.load(f)
                # self._network.convnet.dual_ini(0)
                # self._network.convnet.dual_ini(1)
                self._network.to(self._device)
                return
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            f = open('./ini_resnet18_cifar100', 'wb')
            pickle.dump(self._network, f)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader,syn_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        self._network.to(self._device)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                
                # seed = int(time.time() * 1000) % 100000
                # inputs = DiffAugment(inputs, self.dsa_strategy, seed=seed, param=self.dsa_param)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
        # self._network.convnet.dual_ini(0)
        # self._network.convnet.dual_ini(1)
        logging.info(info)

    def _update_representation(self, train_loader,syn_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            cnn_accy, nme_accy = self.eval_task()
            print(cnn_accy["grouped"])
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                # self._network.convnet.dual_batch(0)
                # self._network.to(self._device)
                seed = int(time.time() * 1000) % 100000
                
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # syn_inputs, syn_targets = self.get_random_batch(batch_size)
                # syn_inputs, syn_targets = syn_inputs.to(self._device), syn_targets.to(self._device)
                #
                # inputs = torch.cat([inputs,syn_inputs])
                # targets = torch.cat([targets,syn_targets])
                #
                # inputs = DiffAugment(inputs, self.dsa_strategy, seed=seed, param=self.dsa_param)

                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = loss_clf + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                # n step
                #self._network.convnet.dual_batch(1)
                self._network.to(self._device)
                
                for i in range(step_syn):
                    seed = int(time.time() * 1000) % 100000
                    inputs_syn, targets_syn = self.get_random_batch(batch_size)
                    # inputs_syn = DiffAugment(inputs_syn, self.dsa_strategy, seed=seed, param=self.dsa_param)
                    inputs_syn, targets_syn = inputs_syn.to(self._device), targets_syn.to(self._device)
                    logits = self._network(inputs_syn)["logits"]

                    loss_clf = F.cross_entropy(logits, targets_syn)
                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes],
                        self._old_network(inputs_syn)["logits"],
                        T,
                    )

                    loss = (loss_clf+loss_kd)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets_syn.expand_as(preds)).cpu().sum()
                    total += len(targets_syn)

            # for i, (_, inputs, targets) in enumerate(syn_loader):
            #     inputs, targets = inputs.to(self._device), targets.to(self._device)
            #
            #     seed = int(time.time() * 1000) % 100000
            #     inputs = DiffAugment(inputs, self.dsa_strategy, seed=seed, param=self.dsa_param)
            #     logits = self._network(inputs)["logits"]
            #
            #     loss_clf = F.cross_entropy(logits, targets)
            #     loss_kd = _KD_loss(
            #         logits[:, : self._known_classes],
            #         self._old_network(inputs)["logits"],
            #         T,
            #     )
            #
            #     loss = (loss_clf+loss_kd)
            #
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            #     losses += loss.item()
            #
            #     _, preds = torch.max(logits, dim=1)
            #     correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            #     total += len(targets)
            #self._network.convnet.dual_batch(0)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _construct_exemplar_synthetic(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} for old classes)".format(m)
        )
        classes_range = np.arange(self._known_classes, self._total_classes)
        data, targets, _ = data_manager.get_dataset(classes_range
            ,
            source="train",
            mode="train",
            ret_data=True,
        )
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        # task3
        # theta2 = (D1+T2,theta1)
        # D1+D2+T3   D2= (theta2,ran), D1 = (theta1,T1)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data = torch.stack([transform(img) for img in data]).numpy()
        # distill_data + task+data
        # real_data = np.concatenate((self._data_memory, data)) if len(self._data_memory) != 0 else data
        real_data = data
        # Select
        # real_label = np.concatenate((self._targets_memory, targets)) if len(self._targets_memory) != 0 else targets
        real_label = targets
        syn_data, syn_lablel = self.dd.gen_synthetic_data(self._old_network,None,real_data,real_label,classes_range)
        syn_data = denormalize_cifar100(syn_data)
        syn_data = tensor2img(syn_data)
        syn_lablel = syn_lablel.cpu().numpy()
        self._data_memory = (
            np.concatenate((self._data_memory, syn_data))
            if len(self._data_memory) != 0
            else syn_data
            )
        self._targets_memory = (
            np.concatenate((self._targets_memory, syn_lablel))
            if len(self._targets_memory) != 0
            else syn_lablel
            )
        # debug: syn_data.shape(10, 32, 32, 3), but should be (100, 32, 32, 3)
        # syn_lablel.shape (10, ), but should be (100,)
        # pdb.set_trace()
        # self._data_memory = (
        #     torch.cat((self._data_memory, syn_data))
        #     if len(self._data_memory) != 0
        #     else syn_data
        #     )
        # self._targets_memory = (
        #     torch.cat((self._targets_memory, syn_lablel))
        #     if len(self._targets_memory) != 0
        #     else syn_lablel
        #     )

        # self._data_memory = syn_data
        # self._targets_memory = syn_lablel


    def build_rehearsal_memory(self, data_manager, per_class,is_dd=True):
        if self._fixed_memory:
            if is_dd:
                self._construct_exemplar_synthetic(data_manager, per_class)
            else:
                self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def get_random_batch(self, batch_size):
        """Returns a random batch according to current valid size."""
        global_bs = batch_size
        # if global batch size > current valid size, we just sample with replacement
        replace = False if len(self._targets_memory) >= global_bs else True

        random_indices = np.random.choice(
            np.arange(len(self._targets_memory)), size=global_bs, replace=replace)
        
        image = self._data_memory[random_indices]
        label = self._targets_memory[random_indices]
        seed = int(time.time() * 1000) % 100000
        # image = DiffAugment(image, self.dsa_strategy, seed=seed, param=self.dsa_param)
        normalize = transforms.Normalize(mean = [0.5071, 0.4866, 0.4409],
        std = [0.2673, 0.2564, 0.2762])
        train_trsf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        normalize
        ])
        data = [train_trsf(Image.fromarray(img)) for img in image]
        image = torch.stack(data)
        return [image, torch.tensor(label)]

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
