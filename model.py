from utils import *
import consts

import uuid

import logging
import random
from collections import OrderedDict
import cv2
import imageio

import torch
import torch.nn as nn
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits_loss
from torch.optim import Adam
from torch.utils.data import DataLoader

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s: [AgeBmiProgressionCAAE] [%(levelname)s]: %(message)s")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_layers = nn.ModuleList()

        def add_conv(module_list, name, in_ch, out_ch, kernel, stride, padding, act_fn):
            return module_list.add_module(
                name,
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride,
                    ),
                    act_fn
                )
            )

        add_conv(self.conv_layers, 'e_conv_1', in_ch=3, out_ch=64, kernel=5, stride=2, padding=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_2', in_ch=64, out_ch=128, kernel=5, stride=2, padding=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_3', in_ch=128, out_ch=256, kernel=5, stride=2, padding=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_4', in_ch=256, out_ch=512, kernel=5, stride=2, padding=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_5', in_ch=512, out_ch=1024, kernel=5, stride=2, padding=2, act_fn=nn.ReLU())

        self.fc_layer = nn.Sequential(
            OrderedDict(
                [
                    ('e_fc_1', nn.Linear(in_features=1024, out_features=consts.NUM_Z_CHANNELS)),
                    ('tanh_1', nn.Tanh())
                ]
            )
        )

    def forward(self, face):
        out = face
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        out = out.flatten(1, -1)
        out = self.fc_layer(out)
        return out


class DiscriminatorZ(nn.Module):
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        dims = (consts.NUM_Z_CHANNELS, consts.NUM_ENCODER_CHANNELS, consts.NUM_ENCODER_CHANNELS // 2,
                consts.NUM_ENCODER_CHANNELS // 4)
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.layers.add_module(
                'dz_fc_%d' % i,
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                )
            )

        self.layers.add_module(
            'dz_fc_%d' % (i + 1),
            nn.Sequential(
                nn.Linear(out_dim, 1),
            )
        )

    def forward(self, z):
        out = z
        for layer in self.layers:
            out = layer(out)
        return out

class DiscriminatorImg(nn.Module):
    def __init__(self):
        super(DiscriminatorImg, self).__init__()
        in_dims = (3, 16 + consts.LABEL_LEN_EXPANDED, 32, 64)
        out_dims = (16, 32, 64, 128)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims), 1):
            self.conv_layers.add_module(
                'dimg_conv_%d' % i,
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU()
                )
            )

        self.fc_layers.add_module(
            'dimg_fc_1',
            nn.Sequential(
                nn.Linear(128 * 8 * 8, 1024),
                nn.LeakyReLU()
            )
        )

        self.fc_layers.add_module(
            'dimg_fc_2',
            nn.Sequential(
                nn.Linear(1024, 1),
            )
        )


        self.fc_layers.add_module(
            'dimg_out',
            nn.Sequential(
                nn.Sigmoid()
            )
        )

    def forward(self, imgs, labels, device):
        out = imgs

        for i, conv_layer in enumerate(self.conv_layers, 1):
            out = conv_layer(out)
            if i == 1:
                labels_tensor = torch.zeros(torch.Size((out.size(0), labels.size(1), out.size(2), out.size(3))), device=device)
                for img_idx in range(out.size(0)):
                    for label in range(labels.size(1)):
                        labels_tensor[img_idx, label, :, :] = labels[img_idx, label]
                out = torch.cat((out, labels_tensor), 1)

        out = out.flatten(1, -1)
        for fc_layer in self.fc_layers[:-1]:

            out = fc_layer(out)

        out_sigmoid = self.fc_layers.dimg_out(out)
        return out, out_sigmoid


class DimgWrapperModel(nn.Module):
    def __init__(self, model):
        super(DimgWrapperModel, self).__init__()
        self.model = model

    def forward(self, imgs, labels, device='cpu'):
        if len(labels.shape):
            labels = labels.squeeze(1)
        return self.model(imgs=imgs, labels=labels, device='cpu')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        num_deconv_layers = 5
        mini_size = 4
        self.fc = nn.Sequential(
            nn.Linear(
                consts.NUM_Z_CHANNELS + consts.LABEL_LEN_EXPANDED,
                consts.NUM_GEN_CHANNELS * mini_size ** 2
            ),
            nn.ReLU()
        )

        self.deconv_layers = nn.ModuleList()

        def add_deconv(name, in_dims, out_dims, kernel, stride, actf):
            self.deconv_layers.add_module(
                name,
                nn.Sequential(
                    easy_deconv(
                        in_dims=in_dims,
                        out_dims=out_dims,
                        kernel=kernel,
                        stride=stride,
                    ),
                    actf
                )
            )

        add_deconv('g_deconv_1', in_dims=(1024, 4, 4), out_dims=(512, 8, 8), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_2', in_dims=(512, 8, 8), out_dims=(256, 16, 16), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_3', in_dims=(256, 16, 16), out_dims=(128, 32, 32), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_4', in_dims=(128, 32, 32), out_dims=(64, 64, 64), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_5', in_dims=(64, 64, 64), out_dims=(32, 128, 128), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_6', in_dims=(32, 128, 128), out_dims=(16, 128, 128), kernel=5, stride=1, actf=nn.ReLU())
        add_deconv('g_deconv_7', in_dims=(16, 128, 128), out_dims=(3, 128, 128), kernel=1, stride=1, actf=nn.Tanh())

    def forward(self, z, age=None, bmi_group=None):
        out = z
        if age is not None and bmi_group is not None:
            label = Label(age, bmi_group).to_tensor() \
                if (isinstance(age, int) and isinstance(bmi_group, int)) \
                else torch.cat((age, bmi_group), 1)
            out = torch.cat((out, label), 1)
        out = self.fc(out)
        out = out.view(out.size(0), 1024, 4, 4)
        for i, deconv_layer in enumerate(self.deconv_layers, 1):
            out = deconv_layer(out)
        return out


class Net(object):
    def __init__(self):
        self.E = Encoder()
        self.Dz = DiscriminatorZ()
        self.Dimg = DiscriminatorImg()
        self.G = Generator()

        self.eg_optimizer = Adam(list(self.E.parameters()) + list(self.G.parameters()))
        self.dz_optimizer = Adam(self.Dz.parameters())
        self.di_optimizer = Adam(self.Dimg.parameters())

        self.device = None
        self.cpu()

    def __call__(self, *args, **kwargs):
        self.test_single(*args, **kwargs)

    def __repr__(self):
        return os.linesep.join([repr(subnet) for subnet in (self.E, self.Dz, self.G)])

    def test_single_internal(self, image_tensor, age, orig_bmi_group, target):
        batch = image_tensor.repeat(consts.NUM_AGES, 1, 1, 1).to(device=self.device)
        z = self.E(batch)

        bmi_group_tensor = -torch.ones(consts.NUM_BMI_GROUPS)
        bmi_group_tensor[int(orig_bmi_group)] *= -1
        bmi_group_tensor = bmi_group_tensor.repeat(consts.NUM_AGES, consts.NUM_AGES // consts.NUM_BMI_GROUPS)

        age_tensor = -torch.ones(consts.NUM_AGES, consts.NUM_AGES)
        for i in range(consts.NUM_AGES):
            age_tensor[i][i] *= -1

        l = torch.cat((age_tensor, bmi_group_tensor), 1).to(self.device)
        z_l = torch.cat((z, l), 1)

        generated = self.G(z_l)

        if True:
            image_tensor = image_tensor.permute(1, 2, 0)
            image_tensor = 255 * one_sided(image_tensor.numpy())
            image_tensor = np.ascontiguousarray(image_tensor, dtype=np.uint8)
            font = cv2.FONT_HERSHEY_PLAIN
            bottomLeftCornerOfText = (1, 10)
            fontScale = .8
            fontColor = (255, 255, 255)
            cv2.putText(
                image_tensor,
                'BMI:{}|Age: {}'.format(["Healthy", "Overweight", "Obese"][orig_bmi_group], age),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
            )
            image_tensor = two_sided(torch.from_numpy(image_tensor / 255.0)).float().permute(2, 0, 1)

        for idx, prediction in enumerate(generated):
            dest = os.path.join(target, f'{idx}.jpg')
            save_image_normalized(tensor=prediction, filename=dest)
            logging.info(f"Saved age progression result to: {os.path.basename(dest)}")

    def test_single(self, image_tensor, image_name, age_group, bmi_group, target):
        self.eval()
        images = []

        images.append(self.test_single_internal(image_tensor=image_tensor,
                                                age=age_group,
                                                orig_bmi_group=bmi_group,
                                                target=target))

    def teach_encoder_generator_discriminatorZ(self, z, z_prior, generated, images, labels, losses,
                                               local_explainable, explanation_type, trained_data):
        self.eg_optimizer.zero_grad()
        self.dz_optimizer.zero_grad()

        d_z_prior = self.Dz(z_prior)
        d_z = self.Dz(z)

        dz_loss_prior = bce_with_logits_loss(d_z_prior, torch.ones_like(d_z_prior))
        dz_loss = bce_with_logits_loss(d_z, torch.zeros_like(d_z))
        dz_loss_tot = (dz_loss + dz_loss_prior)
        losses['dz'].append(dz_loss_tot.item())

        input_output_loss = l1_loss

        eg_loss = input_output_loss(generated, images)
        losses['eg'].append(eg_loss.item())

        reg = l1_loss(generated[:, :, :, :-1], generated[:, :, :, 1:]) + l1_loss(generated[:, :, :-1, :], generated[:, :, 1:, :])

        reg_loss = 0 * reg
        reg_loss.to(self.device)
        losses['reg'].append(reg_loss.item())

        d_z = self.Dz(z)
        ez_loss = 0.0001 * bce_with_logits_loss(d_z, torch.ones_like(d_z))
        ez_loss.to(self.device)
        losses['ez'].append(ez_loss.item())

        d_i_output, d_i_output_sigmoid = self.Dimg(generated, labels, self.device)

        dg_loss = 0.0001 * bce_with_logits_loss(d_i_output, torch.ones_like(d_i_output))
        losses['dg'].append(dg_loss.item())

        loss = eg_loss + reg_loss + ez_loss + dg_loss
        loss.backward(retain_graph=True)

        dz_loss_tot.backward()

        self.eg_optimizer.step()
        self.dz_optimizer.step()

        return loss

    def teach_discriminator_img(self, generated, images, labels, losses):
        self.di_optimizer.zero_grad()

        d_i_input, d_i_input_sigmoid = self.Dimg(images, labels, self.device)
        d_i_output, d_i_output_sigmoid = self.Dimg(generated, labels, self.device)

        di_input_loss = bce_with_logits_loss(d_i_input, torch.ones_like(d_i_input))
        di_output_loss = bce_with_logits_loss(d_i_output, torch.zeros_like(d_i_output))
        di_loss_tot = (di_input_loss + di_output_loss)
        losses['di'].append(di_loss_tot.item())

        di_loss_tot.backward()
        self.di_optimizer.step()

    def teachSplit(
            self,
            dataset_path,
            batch_size=64,
            epochs=1,
            weight_decay=1e-5,
            lr=2e-4,
            should_plot=False,
            betas=(0.9, 0.999),
            valid_size=None,
            where_to_save=None,
            models_saving='always',
            explainable=False,
            explanation_type=None):

        where_to_save = where_to_save or default_where_to_save()
        logging.info(f"Model saving path: {where_to_save}")

        logging.debug(f"Loading dataset from {dataset_path}...")
        dataset = get_dataset(dataset_path)
        valid_size = valid_size or batch_size
        valid_dataset, train_dataset = torch.utils.data.random_split(dataset, (valid_size, len(dataset) - valid_size))

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

        logging.debug(f"Dataset loaded")

        input_output_loss = l1_loss
        nrow = round((2 * batch_size)**0.5)

        for optimizer in (self.eg_optimizer, self.dz_optimizer, self.di_optimizer):
            for param in ('weight_decay', 'betas', 'lr'):
                val = locals()[param]
                if val is not None:
                    optimizer.param_groups[0][param] = val

        loss_tracker = LossTracker(plot=should_plot)
        where_to_save_epoch = ""
        save_count = 0
        paths_for_gif = []

        trained_data = None

        local_explainable = False
        for epoch in range(1, epochs + 1):
            logging.debug(f"Epoch: {epoch}/{epochs+1}")

            where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch))
            try:
                if not os.path.exists(where_to_save_epoch) & epoch % 20 == 0:
                    os.makedirs(where_to_save_epoch)
                paths_for_gif.append(where_to_save_epoch)
                losses = defaultdict(lambda: [])

                self.train()
                for i, (images, labels) in enumerate(train_loader, 1):

                    images = images.to(device=self.device)
                    labels = torch.stack([str_to_tensor(idx_to_class[l], normalize=True) for l in list(labels.numpy())])
                    labels = labels.to(device=self.device)

                    logging.debug(f"\tIteration: {i} images shape: {str(images.shape)}")

                    z = self.E(images)
                    z_l = torch.cat((z, labels), 1)
                    generated = self.G(z_l)

                    z_prior = two_sided(torch.rand_like(z, device=self.device))

                    loss = self.teach_encoder_generator_discriminatorZ(z=z, z_prior=z_prior, generated=generated,
                                                                       images=images, labels=labels, losses=losses,
                                                                       local_explainable=local_explainable,
                                                                       explanation_type=explanation_type,
                                                                       trained_data=trained_data)
                    loss = loss.detach()


                    self.teach_discriminator_img(generated.detach(), images, labels, losses)

                    uni_diff_loss = (uni_loss(z.cpu().detach()) - uni_loss(z_prior.cpu().detach())) / batch_size

                    now = datetime.datetime.now()

                logging.info('[{h}:{m}[Epoch {e}] Loss: {t}'.format(h=now.hour, m=now.minute, e=epoch, t=loss.item()))
                print_timestamp(f"[Epoch {epoch:d}] Loss: {loss.item():f}")
                to_save_models = models_saving in ('always', 'tail')
                if epoch % 20 == 0:
                    cp_path = self.save(where_to_save_epoch, to_save_models=to_save_models)
                else:
                    cp_path = self.save(where_to_save_epoch, to_save_models=False)
                if models_saving == 'tail':
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)

                loss_tracker.save(os.path.join(cp_path, 'losses.png'))


                with torch.no_grad():
                    self.eval()

                    for ii, (images, labels) in enumerate(valid_loader, 1):
                        images = images.to(self.device)
                        labels = torch.stack([str_to_tensor(idx_to_class[l], normalize=True) for l in list(labels.numpy())])
                        labels = labels.to(self.device)
                        validate_labels = labels.to(self.device)

                        z = self.E(images)
                        z_l = torch.cat((z, validate_labels), 1)
                        generated = self.G(z_l)

                        loss = input_output_loss(images, generated)

                        joined = merge_images(images, generated)
                        file_name = os.path.join(where_to_save_epoch, 'validation.png')
                        save_image_normalized(tensor=joined, filename=file_name, nrow=nrow)

                        losses['valid'].append(loss.item())
                        break


                loss_tracker.append_many(**{k: mean(v) for k, v in losses.items()})
                loss_tracker.plot()

                logging.info('[{h}:{m}[Epoch {e}] Loss: {l}'.format(h=now.hour, m=now.minute, e=epoch, l=repr(loss_tracker)))

            except KeyboardInterrupt:
                print_timestamp("{br}CTRL+C detected, saving model{br}".format(br=os.linesep))
                if models_saving != 'never':
                    cp_path = self.save(where_to_save_epoch, to_save_models=True)
                if models_saving == 'tail':
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, 'losses.png'))
                raise

        if models_saving == 'last':
            cp_path = self.save(where_to_save_epoch, to_save_models=True)
        loss_tracker.plot()

    def teach(
            self,
            dataset_path,
            batch_size=64,
            epochs=1,
            weight_decay=1e-5,
            lr=2e-4,
            should_plot=False,
            betas=(0.9, 0.999),
            valid_size=None,
            where_to_save=None,
            models_saving='always',
            explainable=True,
            explanation_type='saliency'):

        where_to_save = where_to_save or default_where_to_save()
        dataset = get_dataset(dataset_path)
        valid_size = valid_size or batch_size
        valid_dataset, train_dataset = torch.utils.data.random_split(dataset, (valid_size, len(dataset) - valid_size))

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

        input_output_loss = l1_loss
        nrow = round((2 * batch_size)**0.5)

        for optimizer in (self.eg_optimizer, self.dz_optimizer, self.di_optimizer):
            for param in ('weight_decay', 'betas', 'lr'):
                val = locals()[param]
                if val is not None:
                    optimizer.param_groups[0][param] = val

        loss_tracker = LossTracker(plot=should_plot)
        where_to_save_epoch = ""
        save_count = 0
        paths_for_gif = []
        trained_data = None

        local_explainable = False
        for epoch in range(1, epochs + 1):
            where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch))
            try:
                if not os.path.exists(where_to_save_epoch) & epoch % 20 == 0:
                    os.makedirs(where_to_save_epoch)
                paths_for_gif.append(where_to_save_epoch)
                losses = defaultdict(lambda: [])

                self.train()
                for i, (images, labels) in enumerate(train_loader, 1):

                    images = images.to(device=self.device)
                    labels = torch.stack([str_to_tensor(idx_to_class[l], normalize=True) for l in list(labels.numpy())])
                    labels = labels.to(device=self.device)
                    z = self.E(images)

                    z_l = torch.cat((z, labels), 1)
                    generated = self.G(z_l)
                    eg_loss = input_output_loss(generated, images)
                    losses['eg'].append(eg_loss.item())

                    reg = l1_loss(generated[:, :, :, :-1], generated[:, :, :, 1:]) + l1_loss(generated[:, :, :-1, :], generated[:, :, 1:, :])

                    reg_loss = 0 * reg
                    reg_loss.to(self.device)
                    losses['reg'].append(reg_loss.item())

                    z_prior = two_sided(torch.rand_like(z, device=self.device))
                    d_z_prior = self.Dz(z_prior)
                    d_z = self.Dz(z)

                    dz_loss_prior = bce_with_logits_loss(d_z_prior, torch.ones_like(d_z_prior))
                    dz_loss = bce_with_logits_loss(d_z, torch.zeros_like(d_z))
                    dz_loss_tot = (dz_loss + dz_loss_prior)
                    losses['dz'].append(dz_loss_tot.item())


                    ez_loss = 0.0001 * bce_with_logits_loss(d_z, torch.ones_like(d_z))
                    ez_loss.to(self.device)
                    losses['ez'].append(ez_loss.item())

                    d_i_input, d_i_input_sigmoid = self.Dimg(images, labels, self.device)
                    d_i_output, d_i_output_sigmoid = self.Dimg(generated.detach(), labels, self.device)
                    di_input_loss = bce_with_logits_loss(d_i_input, torch.ones_like(d_i_input))
                    di_output_loss = bce_with_logits_loss(d_i_output, torch.zeros_like(d_i_output))
                    di_loss_tot = (di_input_loss + di_output_loss)
                    losses['di'].append(di_loss_tot.item())

                    dg_loss = 0.0001 * bce_with_logits_loss(d_i_output, torch.ones_like(d_i_output))
                    losses['dg'].append(dg_loss.item())

                    uni_diff_loss = (uni_loss(z.cpu().detach()) - uni_loss(z_prior.cpu().detach())) / batch_size

                    self.eg_optimizer.zero_grad()
                    loss = eg_loss + reg_loss + ez_loss + dg_loss
                    self.dz_optimizer.zero_grad()
                    self.di_optimizer.zero_grad()

                    loss.backward(retain_graph=True)
                    dz_loss_tot.backward(retain_graph=True)
                    di_loss_tot.backward()

                    self.eg_optimizer.step()
                    self.dz_optimizer.step()
                    self.di_optimizer.step()

                    now = datetime.datetime.now()

                logging.info('[{h}:{m}[Epoch {e}] Loss: {t}'.format(h=now.hour, m=now.minute, e=epoch, t=loss.item()))
                print_timestamp(f"[Epoch {epoch:d}] Loss: {loss.item():f}")
                to_save_models = models_saving in ('always', 'tail')
                if epoch % 20 == 0:
                    cp_path = self.save(where_to_save_epoch, to_save_models=to_save_models)
                else:
                    cp_path = self.save(where_to_save_epoch, to_save_models=False)
                if models_saving == 'tail':
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, 'losses.png'))

                with torch.no_grad():
                    self.eval()

                    for ii, (images, labels) in enumerate(valid_loader, 1):
                        images = images.to(self.device)
                        labels = torch.stack([str_to_tensor(idx_to_class[l], normalize=True) for l in list(labels.numpy())])
                        labels = labels.to(self.device)
                        validate_labels = labels.to(self.device)

                        z = self.E(images)
                        z_l = torch.cat((z, validate_labels), 1)
                        generated = self.G(z_l)

                        loss = input_output_loss(images, generated)

                        joined = merge_images(images, generated)

                        file_name = os.path.join(where_to_save_epoch, 'validation.png')
                        save_image_normalized(tensor=joined, fp=file_name, nrow=nrow)

                        losses['valid'].append(loss.item())
                        break


                loss_tracker.append_many(**{k: mean(v) for k, v in losses.items()})
                loss_tracker.plot()

                logging.info('[{h}:{m}[Epoch {e}] Loss: {l}'.format(h=now.hour, m=now.minute, e=epoch, l=repr(loss_tracker)))

            except KeyboardInterrupt:
                print_timestamp("{br}CTRL+C detected, saving model{br}".format(br=os.linesep))
                if models_saving != 'never':
                    cp_path = self.save(where_to_save_epoch, to_save_models=True)
                if models_saving == 'tail':
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, 'losses.png'))
                raise

        if models_saving == 'last':
            cp_path = self.save(where_to_save_epoch, to_save_models=True)
        loss_tracker.plot()

    def _mass_fn(self, fn_name, *args, **kwargs):
        for class_attr in dir(self):
            if not class_attr.startswith('_'):
                class_attr = getattr(self, class_attr)
                if hasattr(class_attr, fn_name):
                    fn = getattr(class_attr, fn_name)
                    fn(*args, **kwargs)

    def to(self, device):
        self._mass_fn('to', device=device)

    def cpu(self):
        self._mass_fn('cpu')
        self.device = torch.device('cpu')

    def cuda(self):
        self._mass_fn('cuda')
        self.device = torch.device('cuda')

    def eval(self):
        self._mass_fn('eval')

    def train(self):
        self._mass_fn('train')

    def save(self, path, to_save_models=True):
        if not os.path.isdir(path):
            os.mkdir(path)
        if not os.path.isdir(path):
            os.mkdir(path)

        saved = []
        if to_save_models:
            for class_attr_name in dir(self):
                if not class_attr_name.startswith('_'):
                    class_attr = getattr(self, class_attr_name)
                    if hasattr(class_attr, 'state_dict'):
                        state_dict = class_attr.state_dict
                        fname = os.path.join(path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name))
                        torch.save(state_dict, fname)
                        saved.append(class_attr_name)

        if saved:
            logging.info(f"Saved age progression image {os.path.basename(path)}")
        elif to_save_models:
            raise FileNotFoundError("Nothing was saved to {}".format(path))
        return path

    def load(self, path, slim=True):
        loaded = []
        for class_attr_name in dir(self):
            if (not class_attr_name.startswith('_')) and ((not slim) or (class_attr_name in ('E', 'G'))):
                class_attr = getattr(self, class_attr_name)
                fname = os.path.join(path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name))
                if hasattr(class_attr, 'load_state_dict') and os.path.exists(fname):
                    class_attr.load_state_dict(torch.load(fname, map_location=torch.device('cpu'))())
                    loaded.append(class_attr_name)
        if loaded:
            print_timestamp("Loaded {} from {}".format(', '.join(loaded), path))
        else:
            raise FileNotFoundError("Nothing was loaded from {}".format(path))

def create_list_of_img_paths(pattern, start, step):
    result = []
    fname = pattern.format(start)
    while os.path.isfile(fname):
        result.append(fname)
        start += step
        fname = pattern.format(start)
    return result