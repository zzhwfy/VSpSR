import torch
import torch.nn as nn
import torchvision


class VSPSR(nn.Module):
    def __init__(self, vspm, postprocess=None):
        super(VSPSR, self).__init__()
        self.vspm = vspm

        if postprocess is not None:
            self.postprocess = postprocess
        else:
            self.postprocess = None

    def forward(self, x):
        result_dict = {}
        e, kl_z, kl_w, explore = self.vspm(x)
        output = e
        result_dict.update({'kl_z': kl_z, 'kl_w': kl_w, 'e': explore})

        if self.postprocess is not None:
            output = self.postprocess(output, x)
        result_dict.update({'pred': output})

        return result_dict


class Discriminator(nn.Module):
    def __init__(self, in_chans=3):
        super(Discriminator, self).__init__()
        self.in_chans = in_chans
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_chans, 64, 4, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.Conv2d(512, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return torch.mean(x, dim=(1, 2, 3))


class TruncatedVGG19(nn.Module):
    """
    truncated VGG19, to calculate MSE loss in the VGG space
    """

    def __init__(self, i, j):
        super(TruncatedVGG19, self).__init__()

        vgg19 = torchvision.models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        for layer in vgg19.features.children():
            truncate_at += 1

            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            if maxpool_counter == i - 1 and conv_counter == j:
                break

        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        input: HR or SR，size is (N, 3, w * scaling factor, h * scaling factor)
        output: VGG19 features，size is (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)
        return output


class SetCriterion(nn.Module):
    """
    This class computes the loss for Super-Resolution.
    """

    def __init__(self, losses, weight_dict, args):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        self.args = args
        if args.GAN:
            self.Discriminator = Discriminator()
        if args.VGG:
            self.truncated_vgg19 = TruncatedVGG19(i=args.VGG_i, j=args.VGG_j)
        self.MSE = torch.nn.MSELoss()

    def loss_MSE(self, outputs, hr):
        """
        Compute the MSE loss
        """
        pred = outputs["pred"]
        losses = {
            "loss_MSE": self.MSE(pred, hr),
        }
        return losses

    def loss_KL_z(self, outputs, hr):
        kl_z = outputs["kl_z"]
        losses = {
            "loss_KL_z": torch.mean(kl_z)
        }

        return losses

    def loss_KL_w(self, outputs, hr):
        kl_w = outputs["kl_w"]
        losses = {
            "loss_KL_w": torch.mean(kl_w)
        }
        return losses

    def loss_G(self, outputs, hr):
        pred = outputs["pred"]
        real = self.Discriminator(pred)
        label = torch.ones_like(real)
        losses = {
            "loss_G": self.MSE(real, label)
        }
        return losses

    def loss_D(self, outputs, hr):
        pred = outputs["pred"].detach()
        real = self.Discriminator(hr)
        real_label = torch.ones_like(real)
        fake = self.Discriminator(pred)
        fake_label = torch.zeros_like(fake)
        losses = {
            "loss_D": 0.5 * self.MSE(fake, fake_label) + 0.5 * self.MSE(real, real_label)
        }
        return losses

    def loss_C(self, outputs, hr):
        pred = outputs["pred"]
        pred_in_vgg_space = self.truncated_vgg19(pred)
        hr_in_vgg_space = self.truncated_vgg19(hr)
        losses = {
            "loss_C": self.MSE(pred_in_vgg_space, hr_in_vgg_space)
        }
        return losses

    def get_loss(self, loss, outputs, hr):
        loss_map = {'KL_z': self.loss_KL_z,
                    'KL_w': self.loss_KL_w,
                    'MSE': self.loss_MSE,
                    'G': self.loss_G,
                    'D': self.loss_D,
                    'C': self.loss_C}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, hr)

    def forward(self, outputs, hr):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, hr))
        return losses
