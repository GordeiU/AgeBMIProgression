from tqdm import tqdm
import os
import datetime
from shutil import copyfile
from collections import namedtuple, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.metrics import mean_squared_error as mse

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import consts
import matplotlib
matplotlib.use('Agg')

def save_image_normalized(*args, **kwargs):
    save_image(*args, **kwargs, normalize=True, range=(-1, 1), padding=4)


def two_sided(x):
    return 2 * (x - 0.5)

def one_sided(x):
    return (x + 1) / 2


pil_to_model_tensor_transform = transforms.Compose(
    [
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.mul(2).sub(1))  # Tensor elements domain: [0:1] -> [-1:1]
    ]
)


def get_dataset(root):
    logging.info(f"Getting dataset from: {root}")
    ret = lambda: ImageFolder(os.path.join(root, 'labeled'), transform=pil_to_model_tensor_transform)
    try:
        return ret()
    except (RuntimeError, FileNotFoundError):
        sort_to_classes(os.path.join(root, 'unlabeled'))
        return ret()

def sort_to_classes(root):
    def log(text):
        logging.info(f"[Labeling process] {text}")

    log('Starting labeling process...')
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    if not files:
        raise FileNotFoundError(f"No image files in {root}")

    copied_count = 0
    sorted_folder = os.path.join(root, '..', 'labeled')

    if not os.path.isdir(sorted_folder):
        os.mkdir(sorted_folder)

    for f in tqdm(files):
        matcher = consts.ORIGINAL_IMAGE_FORMAT.match(f)
        if matcher is None:
            continue

        id, age, bmi = matcher.groups()

        srcfile = os.path.join(root, f)
        label = Label(int(age), int(bmi))
        dstfolder = os.path.join(sorted_folder, label.to_str())
        dstfile = os.path.join(dstfolder, id+'.jpg')
        if os.path.isfile(dstfile):
            continue
        if not os.path.isdir(dstfolder):
            os.mkdir(dstfolder)
        copyfile(srcfile, dstfile)
        copied_count += 1

    log('Finished labeling process')

def str_to_tensor(text, normalize=False):
    age_group, bmi_group = text.split('.')
    age_tensor = -torch.ones(consts.NUM_AGES)
    age_tensor[int(age_group)] *= -1
    bmi_group_tensor = -torch.ones(consts.NUM_BMI_GROUPS)
    bmi_group_tensor[int(bmi_group)] *= -1
    if normalize:
        bmi_group_tensor = bmi_group_tensor.repeat(consts.NUM_AGES // consts.NUM_BMI_GROUPS)
    result = torch.cat((age_tensor, bmi_group_tensor), 0)
    return result


class Label(namedtuple('Label', ('age', 'bmi_group'))):
    def __init__(self, age, bmi_group):
        super(Label, self).__init__()
        self.age_group = self.age_transform(self.age)

    def to_str(self):
        return '%d.%d' % (self.age_group, self.bmi_group)

    @staticmethod
    def age_transform(age): #TODO: Modify to fit my needs
        age -= 1
        if age < 20:
            # first 4 age groups are for kids <= 20, 5 years intervals
            return max(age // 5, 0)
        else:
            # last (6?) age groups are for adults > 20, 10 years intervals
            return min(4 + (age - 20) // 10, consts.NUM_AGES - 1)

    def to_tensor(self, normalize=False):
        return str_to_tensor(self.to_str(), normalize=normalize)



fmt_t = "%H_%M"
fmt = "%Y_%m_%d"

def default_train_results_dir():
    return os.path.join('.', 'trained_models', datetime.datetime.now().strftime(fmt), datetime.datetime.now().strftime(fmt_t))


def default_where_to_save(eval=True):
    path_str = os.path.join('.', 'results', datetime.datetime.now().strftime(fmt), datetime.datetime.now().strftime(fmt_t))
    if not os.path.exists(path_str):
        os.makedirs(path_str)


def default_test_results_dir(eval=True):
    return os.path.join('.', 'test_results', datetime.datetime.now().strftime(fmt) if eval else fmt)


def print_timestamp(s):
    logging.debug(f"[{datetime.datetime.now().strftime(fmt_t.replace('_', ':'))}] {x}")

class LossTracker(object):
    def __init__(self, use_heuristics=False, plot=False, eps=1e-3):
        # assert 'train' in names and 'valid' in names, str(names)
        self.losses = defaultdict(lambda: [])
        self.paths = []
        self.epochs = 0
        self.use_heuristics = use_heuristics
        if plot:
            plt.ion()
            plt.show()
        else:
            plt.switch_backend("agg")

    # deprecated
    # def append(self, train_loss, valid_loss, tv_loss, uni_loss, path):
    #     self.train_losses.append(train_loss)
    #     self.valid_losses.append(valid_loss)
    #     self.tv_losses.append(tv_loss)
    #     self.uni_losses.append(uni_loss)
    #     self.paths.append(path)
    #     self.epochs += 1
    #     if self.use_heuristics and self.epochs >= 2:
    #         delta_train = self.train_losses[-1] - self.train_losses[-2]
    #         delta_valid = self.valid_losses[-1] - self.valid_losses[-2]
    #         if delta_train < -self.eps and delta_valid < -self.eps:
    #             pass  # good fit, continue training
    #         elif delta_train < -self.eps and delta_valid > +self.eps:
    #             pass  # overfit, consider stop the training now
    #         elif delta_train > +self.eps and delta_valid > +self.eps:
    #             pass  # underfit, if this is in an advanced epoch, break
    #         elif delta_train > +self.eps and delta_valid < -self.eps:
    #             pass  # unknown fit, check your model, optimizers and loss functions
    #         elif 0 < delta_train < +self.eps and self.epochs >= 3:
    #             prev_delta_train = self.train_losses[-2] - self.train_losses[-3]
    #             if 0 < prev_delta_train < +self.eps:
    #                 pass  # our training loss is increasing but in less than eps,
    #                 # this is a drift that needs to be caught, consider lower eps next time
    #         else:
    #             pass  # saturation \ small fluctuations

    def append_single(self, name, value):
        self.losses[name].append(value)

    def append_many(self, **names):
        for name, value in names.items():
            self.append_single(name, value)

    def append_many_and_plot(self, **names):
        self.append_many(**names)

    def plot(self):
        logging.debug("Inside plot")
        plt.clf()
        graphs = [plt.plot(loss, label=name)[0] for name, loss in self.losses.items()]
        plt.legend(handles=graphs)
        plt.xlabel('Epochs')
        plt.ylabel('Averaged loss')
        plt.title('Losses by epoch')
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

    @staticmethod
    def show():
        logging.info("Inside show")
        plt.show()

    @staticmethod
    def save(path):
        plt.savefig(path, transparent=True)

    def __repr__(self):
        ret = {}
        for name, value in self.losses.items():
            ret[name] = value[-1]
        return str(ret)


def mean(l):
    return np.array(l).mean()


def uni_loss(input):
    assert len(input.shape) == 2
    batch_size, input_size = input.size()
    hist = torch.histc(input=input, bins=input_size, min=-1, max=1)
    return mse(hist, batch_size * torch.ones_like(hist)) / input_size


def easy_deconv(in_dims, out_dims, kernel, stride=1, groups=1, bias=True, dilation=1):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)

    c_in, h_in, w_in = in_dims
    c_out, h_out, w_out = out_dims

    padding = [0, 0]
    output_padding = [0, 0]

    lhs_0 = -h_out + (h_in - 1) * stride[0] + kernel[0]  # = 2p[0] - o[0]
    if lhs_0 % 2 == 0:
        padding[0] = lhs_0 // 2
    else:
        padding[0] = lhs_0 // 2 + 1
        output_padding[0] = 1

    lhs_1 = -w_out + (w_in - 1) * stride[1] + kernel[1]  # = 2p[1] - o[1]
    if lhs_1 % 2 == 0:
        padding[1] = lhs_1 // 2
    else:
        padding[1] = lhs_1 // 2 + 1
        output_padding[1] = 1

    return torch.nn.ConvTranspose2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=kernel,
        stride=stride,
        padding=tuple(padding),
        output_padding=tuple(output_padding),
        groups=groups,
        bias=bias,
        dilation=dilation
    )


def remove_trained(folder):
    if os.path.isdir(folder):
        removed_ctr = 0
        for tm in os.listdir(folder):
            tm = os.path.join(folder, tm)
            if os.path.splitext(tm)[1] == consts.TRAINED_MODEL_EXT:
                try:
                    os.remove(tm)
                    removed_ctr += 1
                except OSError as e:
                    logging.error("Failed removing {}: {}".format(tm, e))
        if removed_ctr > 0:
            logging.info("Removed {} trained models from {}".format(removed_ctr, folder))


def merge_images(batch1, batch2):
    assert batch1.shape == batch2.shape
    merged = torch.zeros(batch1.size(0) * 2, batch1.size(1), batch1.size(2), batch1.size(3), dtype=batch1.dtype)
    for i, (image1, image2) in enumerate(zip(batch1, batch2)):
        merged[2 * i] = image1
        merged[2 * i + 1] = image2
    return merged
