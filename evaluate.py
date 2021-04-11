import argparse
import os
import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms

from models import build_model
import utils


def test_for_a_sample(model, img_path, scale_factor=4, gt_path=None, number=1, output_dir=None, filename=None, ii=0):
    img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
    img = img.to(device)
    w = img.shape[2] * scale_factor
    h = img.shape[1] * scale_factor
    n_color = img.shape[0]
    if gt_path is not None:
        gt = transforms.ToTensor()(Image.open(gt_path).convert('RGB'))
        gt = gt.to(device)
        gt_h, gt_w = gt.shape[-2:]

    psnrs = []

    with torch.no_grad():
        assert model is not None
        for i in range(number):
            pred = model(img.unsqueeze(0))["pred"]
            pred = pred.clamp(0, 1).squeeze(0)
            if output_dir is not None:
                filename = filename.split('.')[0]
                output_path = os.path.join(output_dir, f'{ii:06d}_sample{i:05d}.png')
                transforms.ToPILImage()(pred).save(output_path)
            if gt_path is not None:
                # modcrop
                gt = gt[:, :(h // 8) * 8, :(w // 8) * 8]
                _, h1, w1 = gt.shape
                pred = pred[:, :h1, :w1]
                psnr = utils.calc_psnr(pred.to(device), gt)
                print(filename + 'sample{:05d}: psnr: {:.4f}'.format(i, psnr))
                psnrs.append(psnr.item())
    if len(psnrs) > 0:
        print('avg psnr: {:.4f}'.format(np.mean(psnrs)))
        return psnrs
    else:
        return [np.float('inf')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/home/zmic/NTRIE2021SR/DIV2K-te_4X', help='the lr images\' directory')
    parser.add_argument('--output_dir', default='./output', help='the directory to store sr images')
    parser.add_argument('--gt_dir', default=None, help='the ground truth\'s directory')
    parser.add_argument('--first_k', type=int, default=100, help='calculate the first k of samples')
    parser.add_argument('--number', type=int, default=10, help='the number of SR images for every lr image')
    parser.add_argument('--model', default='./checkpoint/checkpoint_x4.pth', help='path to trained model')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--device', default='cuda', type=str, help='device to use')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    global device
    device = torch.device(args.device)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    utils.ensure_path(args.output_dir, remove=False)

    if args.model is not None:
        checkpoint = torch.load(args.model, map_location='cpu')
        model_args = checkpoint['args']
        model_args.postprocess = True
        model, criterion = build_model(model_args)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.to(device)
        model.eval()
    else:
        model = None

    scale = args.scale
    number = args.number
    print('scale factor: {}'.format(scale))

    input_dir = args.input_dir
    gt_dir = args.gt_dir
    output_dir = args.output_dir
    filenames = sorted(os.listdir(input_dir))
    psnr_list = []
    i = 0
    for filename in filenames:
        img_path = os.path.join(input_dir, filename)
        gt_path = os.path.join(gt_dir, filename) if gt_dir is not None else None
        if i + 1 > args.first_k:
            break
        psnr = test_for_a_sample(model, img_path, scale, gt_path, number, output_dir, filename, i)
        psnr_list += psnr
        i = i + 1
    psnr = np.mean(psnr_list)
    print('val: psnr={:.4f}'.format(psnr))
