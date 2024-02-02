import os
import numpy as np
import torch
import  json
from enum import Enum

class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def save_fc(args, model):
    _path = os.path.join(args['logfilename'], "fc.pt")
    if len(args['device']) > 1: 
        fc_weight = model._network.fc.weight.data    
    else:
        fc_weight = model._network.fc.weight.data.cpu()
    torch.save(fc_weight, _path)

    _save_dir = os.path.join(f"./results/fc_weights/{args['prefix']}")
    os.makedirs(_save_dir, exist_ok=True)
    _save_path = os.path.join(_save_dir, f"{args['csv_name']}.csv")
    with open(_save_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},{_path} \n")

def save_model(args, model):
    #used in PODNet
    _path = os.path.join(args['logfilename'], "model.pt")
    if len(args['device']) > 1:
        weight = model._network   
    else:
        weight = model._network.cpu()
    torch.save(weight, _path)


def denormalize_cifar100(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], dtype=np.float16)
        std = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404], dtype=np.float16)
    else:
        mean = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
        std = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor

from PIL import Image
def tensor2img(images,is_batch = True):
    if is_batch:
        image_np = images.cpu().numpy().transpose((0, 2, 3, 1))
    else:
        image_np = images.cpu().numpy().transpose((1, 2, 0))
    pil_image =(image_np* 255).astype(np.uint8)
    return pil_image

