from .vspsr import VSPSR, SetCriterion
from .vspm import make_vspm
from .cem_postprocess import get_cem


def build_loss(args, losses):
    weight_dict = {'loss_KL_z': args.kl_weight,
                   'loss_KL_w': args.kl_weight,
                   'loss_MSE': 1.0,
                   'loss_G': args.G_weight,
                   'loss_C': args.C_weight}
    return SetCriterion(losses=losses, weight_dict=weight_dict, args=args)


def build_model(args):
    vspm = make_vspm(scale=args.scale, n_basis=args.n_basis, alpha=args.alpha, beta=args.beta,
                     variational_z=args.variational_z, variational_w=args.variational_w,
                     upsample=args.upsample, mode=args.upsample_mode)
    if args.postprocess:
        CEM_postprocess = get_cem(args.scale)
    else:
        CEM_postprocess = None

    # LOSS
    if args.postprocess:
        losses = []
    else:
        losses = ['MSE']
    if args.variational_w:
        losses.append('KL_w')
    if args.variational_z:
        losses.append('KL_z')
    if args.GAN:
        losses.append('G')
        losses.append('D')
    if args.VGG:
        losses.append('C')
    criterion = build_loss(args, losses)

    return VSPSR(vspm, CEM_postprocess), criterion
