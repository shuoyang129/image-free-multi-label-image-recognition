import os
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from models import build_model
from utils.validations_baseline import validate_baseline
from opts import arg_parser
from dataloaders import build_dataset
from utils.build_cfg import setup_cfg
import json


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cfg = setup_cfg(args)

    test_split = cfg.DATASET.TEST_SPLIT

    test_dataset = build_dataset(cfg, test_split)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        shuffle=cfg.DATALOADER.TEST.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True)
    classnames = test_dataset.classnames

    model, arch_name = build_model(cfg, args, classnames)

    model.eval()

    if args.pretrained is not None and os.path.exists(args.pretrained):
        print('... loading pretrained weights from %s' % args.pretrained)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict']
        epoch = checkpoint['epoch']
        model.load_state_dict(state_dict, strict=False)
        print('Epoch: %d' % epoch)
    else:
        raise ValueError(
            'args.pretrained is missing or its path does not exist')

    print('Evaluate with threshold %.2f' % args.thre)
    F_1, mAP_score = validate_baseline(
        test_loader, model, args, cfg.DATASET.NAME, logdir=os.path.dirname(args.pretrained))

    print(
            'Test: [{}/{}]\t '
            ' F_1 {:.2f} \t mAP {:.2f}'
            .format(epoch, cfg.OPTIM.MAX_EPOCH,
                    F_1, mAP_score),
            flush=True)


if __name__ == '__main__':
    main()