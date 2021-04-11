from CEM.CEMnet import *
import torch
import torch.nn as nn


class CEM_Post(nn.Module):
    def __init__(self, CEMnet):
        super(CEM_Post, self).__init__()
        self.ds_factor = CEMnet.ds_factor
        self.conf = CEMnet.conf
        inv_hTh_padding = np.floor(np.array(CEMnet.inv_hTh.shape) / 2).astype(np.int32)
        Replication_Padder = nn.ReplicationPad2d(
            (inv_hTh_padding[1], inv_hTh_padding[1], inv_hTh_padding[0], inv_hTh_padding[0]))
        self.Conv_LR_with_Inv_hTh_OP = Filter_Layer(CEMnet.inv_hTh, pre_filter_func=Replication_Padder)
        downscale_antialiasing = np.rot90(CEMnet.ds_kernel, 2)
        upscale_antialiasing = CEMnet.ds_kernel * CEMnet.ds_factor ** 2
        pre_stride, post_stride = calc_strides(None, CEMnet.ds_factor)
        Upscale_Padder = lambda x: nn.functional.pad(x, (
            pre_stride[1], post_stride[1], 0, 0, pre_stride[0], post_stride[0]))
        Aliased_Upscale_OP = lambda x: Upscale_Padder(x.unsqueeze(4).unsqueeze(3)).view(
            [x.size()[0], x.size()[1], CEMnet.ds_factor * x.size()[2], CEMnet.ds_factor * x.size()[3]])
        antialiasing_padding = np.floor(np.array(CEMnet.ds_kernel.shape) / 2).astype(np.int32)
        antialiasing_Padder = nn.ReplicationPad2d(
            (antialiasing_padding[1], antialiasing_padding[1], antialiasing_padding[0], antialiasing_padding[0]))
        self.Upscale_OP = Filter_Layer(upscale_antialiasing,
                                       pre_filter_func=lambda x: antialiasing_Padder(Aliased_Upscale_OP(x)))
        Reshaped_input = lambda x: x.view([x.size()[0], x.size()[1], int(x.size()[2] / self.ds_factor), self.ds_factor,
                                           int(x.size()[3] / self.ds_factor), self.ds_factor])
        Aliased_Downscale_OP = lambda x: Reshaped_input(x)[:, :, :, pre_stride[0], :, pre_stride[1]]
        self.DownscaleOP = Filter_Layer(downscale_antialiasing, pre_filter_func=antialiasing_Padder,
                                        post_filter_func=lambda x: Aliased_Downscale_OP(x))
        self.LR_padder = CEMnet.LR_padder
        self.HR_padder = CEMnet.HR_padder
        self.HR_unpadder = CEMnet.HR_unpadder
        self.LR_unpadder = CEMnet.LR_unpadder  # Debugging tool
        self.pre_pad = False  # Using a variable as flag because I couldn't pass it as argument to forward function when using the DataParallel module with more than 1 GPU
        self.return_2_components = 'decomposed_output' in self.conf.__dict__ and self.conf.decomposed_output

    def forward(self, generated_image, x):
        return_2_components = self.return_2_components and not self.pre_pad
        x = x[:, -3:, :, :]  # Handling the case of adding noise channel(s) - Using only last 3 image channels
        assert np.all(np.mod(generated_image.size()[2:], self.ds_factor) == 0)
        ortho_2_NS_HR_component = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP(x))
        ortho_2_NS_generated = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP(self.DownscaleOP(generated_image)))
        NS_HR_component = generated_image - ortho_2_NS_generated
        if self.conf.sigmoid_range_limit:
            NS_HR_component = torch.tanh(NS_HR_component) * (self.conf.input_range[1] - self.conf.input_range[0])
        output = [ortho_2_NS_HR_component,
                  NS_HR_component] if return_2_components else ortho_2_NS_HR_component + NS_HR_component
        return output

    def train(self, mode=True):
        super(CEM_Post, self).train(mode=mode)
        self.pre_pad = not mode

    def Image_2_Sigmoid_Range_Converter(self, images, opposite_direction=False):
        if opposite_direction:
            return images * (self.conf.input_range[1] - self.conf.input_range[0]) + self.conf.input_range[0]
        else:
            images = torch.clamp(images, min=self.conf.input_range[0], max=self.conf.input_range[1])
            return (images - self.conf.input_range[0]) / (self.conf.input_range[1] - self.conf.input_range[0])

    def Inverse_Sigmoid(self, images):
        return torch.log(
            self.Image_2_Sigmoid_Range_Converter(images) / (1. - self.Image_2_Sigmoid_Range_Converter(images)))


def get_cem(scale=4):
    CEM_conf = Get_CEM_Conf(scale)
    CEM_conf.lower_magnitude_bound = 0.1
    CEM = CEMnet(CEM_conf, upscale_kernel='cubic')
    CEM.WrapArchitecture_PyTorch(only_padders=True)
    CEM_postprocess = CEM_Post(CEM)
    return CEM_postprocess
