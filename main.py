import os
import argparse
import random
import utils
from pathlib import Path
from dataset import build_dataset
from models import build_model
import numpy as np
import torch
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser('Set VSpSR', add_help=False)

    # training setting
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--output_dir', default='logs', help='path where to save logs')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    # dataset setting
    parser.add_argument('--dataset_root', default='NTIRE2021', type=str)
    parser.add_argument('--dataset', default='div2k', type=str)
    parser.add_argument('--crop_size', default=48, type=int)
    parser.add_argument('--scale', default=4, type=int)

    # kbem parameters
    parser.add_argument('--variational_z', action='store_true',
                        help='whether to use variational basis')
    parser.add_argument('--variational_w', action='store_true',
                        help='whether to use variational weight')
    parser.add_argument('--alpha', default=3.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--n_basis', default=256, type=int, help="number of basis")
    parser.add_argument('--kl_weight', default=0.1, type=float, help="weight of KL loss")
    parser.add_argument('--upsample', action='store_true',
                        help='whether to upsample lr')
    parser.add_argument('--upsample_mode', default='bilinear', choices=['bilinear', 'bicubic', 'nearest'])

    # GAN parameters
    parser.add_argument('--GAN', action='store_true',
                        help='whether to use adversarial learning')
    parser.add_argument('--G_weight', default=0.1, type=float, help="weight of generative loss")

    # Perceptual parameters
    parser.add_argument('--VGG', action='store_true',
                        help='whether to use content loss')
    parser.add_argument('--VGG_i', default=5, type=int)
    parser.add_argument('--VGG_j', default=4, type=int)
    parser.add_argument('--C_weight', default=0.01, type=float, help="weight of content loss")

    parser.add_argument('--postprocess', action='store_true',
                        help='whether to use postprocess')

    return parser


def prepare_training(args):
    model, criterion = build_model(args)
    log('vspm: #params={}'.format(utils.compute_num_params(model.vspm, text=True)))
    log(model.vspm)
    log(criterion)
    log('Building dataset...')
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    log('Number of training images: {}'.format(len(dataset_train)))
    log('Number of validation images: {}'.format(len(dataset_val)))

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)

    param_dicts_G = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer_G = torch.optim.Adam(param_dicts_G, lr=args.lr,
                                   weight_decay=args.weight_decay)
    if args.GAN:
        param_dicts_D = [{"params": [p for n, p in criterion.named_parameters() if p.requires_grad]}]
        optimizer_D = torch.optim.Adam(param_dicts_D, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer_D = None

    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, args.lr_drop)
    if optimizer_D is not None:
        lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, args.lr_drop)
    else:
        lr_scheduler_D = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer_G.load_state_dict(checkpoint['optimizer'])
            lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    return model, criterion, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, dataloader_train, dataloader_val


def train_one_epoch(model, criterion, optimizer_G, optimizer_D, dataloader, device, epoch, max_norm):
    model.train()
    criterion.train()
    loss = {'loss_MSE': utils.Averager(),
            'loss_KL_z': utils.Averager(),
            'loss_KL_w': utils.Averager(),
            'loss_total': utils.Averager(),
            'loss_G': utils.Averager(),
            'loss_D': utils.Averager(),
            'loss_C': utils.Averager()}
    log('Epoch: [{}]'.format(epoch))
    print_freq = 10
    total_steps = len(dataloader)
    iterats = iter(dataloader)

    for step in range(total_steps):
        lr, hr = next(iterats)
        lr = lr.to(device)
        hr = hr.to(device)

        outputs = model(lr)
        loss_dict = criterion(outputs, hr)
        weight_dict = criterion.weight_dict
        losses_total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss['loss_total'].add(losses_total.item())
        for k in loss_dict.keys():
            loss[k].add(loss_dict[k].item())

        optimizer_G.zero_grad()
        losses_total.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer_G.step()
        if optimizer_D is not None:
            loss_D = loss_dict["loss_D"]
            optimizer_D.zero_grad()
            loss_D.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm)
            optimizer_D.step()

        if step % print_freq == 0:
            print('step {}/{}:'.format(step, total_steps))
            for k, v in loss_dict.items():
                print('{}:{:.4f}'.format(k, v.item()))

    return loss


@torch.no_grad()
def evaluate(model, criterion, dataloader, device, visualizer, epoch, writer):
    model.eval()
    criterion.eval()
    loss = {'loss_MSE': utils.Averager(),
            'loss_KL_z': utils.Averager(),
            'loss_KL_w': utils.Averager(),
            'loss_total': utils.Averager(),
            'loss_G': utils.Averager(),
            'loss_D': utils.Averager(),
            'loss_C': utils.Averager(),
            'psnr': utils.Averager()}
    total_steps = len(dataloader)
    iterats = iter(dataloader)
    lr_list, hr_list, output_list, e_list = [], [], [], []
    for step in range(total_steps):
        lr, hr = next(iterats)
        lr = lr.to(device)
        hr = hr.to(device)

        outputs = model(lr)
        loss_dict = criterion(outputs, hr)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss['loss_total'].add(losses.item())
        for k in loss_dict.keys():
            loss[k].add(loss_dict[k].item())
        loss['psnr'].add(utils.calc_psnr(outputs['pred'], hr))

        if step + 2 == total_steps:
            lr_list.append(lr)
            output_list.append(outputs['pred'].clamp(0, 1))
            hr_list.append(hr)
            if 'e' in outputs.keys():
                e = (outputs['e'] - outputs['e'].min()) / (outputs['e'].max() - outputs['e'].min() + 1e-6)
                e_list.append(e)

    n = int(np.floor(np.sqrt(hr.size(0))))
    output = outputs['pred'].clamp(0, 1)
    hr = hr[:n * n]
    output = output[:n * n]
    lr = lr[:n * n]
    hr = utils.tile_image(hr, n)
    output = utils.tile_image(output, n)
    lr = utils.tile_image(lr, n)

    visualizer(lr, output, hr, epoch, writer)
    if 'e' in outputs.keys():
        e = (outputs['e'] - outputs['e'].min()) / (outputs['e'].max() - outputs['e'].min() + 1e-6)
        e = e[: n * n]
        e = utils.tile_image(e, n)
        visualizer.save_image(e, 'explore', epoch, writer)

    return loss


def main(args):
    global log, writer
    log, writer = utils.set_save_path(args.output_dir)
    log(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # prepare: model/visualizer/dataset
    visualizer = utils.Visualization()
    model, criterion, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, dataloader_train, dataloader_val = prepare_training(
        args)
    model.to(device)
    criterion.to(device)

    if args.eval:
        test_stats = evaluate(model, criterion, dataloader_val, device, visualizer, 0, writer)

    # training
    output_dir = Path(args.output_dir)
    log("Start training")
    timer = utils.Timer()
    for epoch in range(args.start_epoch, args.epochs + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, args.epochs)]
        writer.add_scalar('lr', optimizer_G.param_groups[0]['lr'], epoch)

        train_stats = train_one_epoch(model, criterion, optimizer_G, optimizer_D, dataloader_train, device, epoch,
                                      args.clip_max_norm)
        lr_scheduler_G.step()
        if lr_scheduler_D is not None:
            lr_scheduler_D.step()
        log_info.append('train:')
        log_info = log_info + ['{}={:.4f}'.format(k, v.item()) for k, v in train_stats.items()]

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                sv_file = {
                    'model': model.state_dict(),
                    'optimizer': optimizer_G.state_dict(),
                    'lr_scheduler': lr_scheduler_G.state_dict(),
                    'epoch': epoch,
                    'args': args
                }
                torch.save(sv_file, checkpoint_path)

        test_stats = evaluate(
            model, criterion, dataloader_val, device, visualizer, epoch, writer
        )
        log_info.append('eval:')
        log_info = log_info + ['{}={:.4f}'.format(k, v.item()) for k, v in test_stats.items()]

        writer.add_scalars('loss', {'train': train_stats['loss_total'].item(),
                                    'eval': test_stats['loss_total'].item()}, epoch)
        writer.add_scalars('psnr', {'val': test_stats['psnr'].item()}, epoch)

        t = timer.t()
        prog = (epoch - args.start_epoch + 1) / (args.epochs - args.start_epoch + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VSpSR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)
