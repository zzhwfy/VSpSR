from .div2k import build as build_div2k


def build_dataset(image_set, args):
    if args.dataset == 'div2k':
        return build_div2k(image_set, args)
    raise ValueError(f'dataset {args.dataset} not supported')
