# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from actionformer.libs.core import load_config
from actionformer.libs.datasets import make_dataset, make_data_loader
from actionformer.libs.modeling import make_meta_arch
from actionformer.libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                                     save_checkpoint, make_optimizer, make_scheduler,
                                     fix_random_seed, ModelEma)


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    cfg['dataset']['backbone'] = args.backbone
    cfg['dataset']['feat_folder'] = args.feat_folder
    cfg['dataset']['num_frames'] = args.num_frames
    cfg['dataset']['feat_stride'] = args.stride
    cfg['dataset']['division_type'] = args.division_type

    json_file_path = cfg['dataset']['json_file']
    json_file_dir = os.path.dirname(json_file_path)
    json_file_name = os.path.basename(json_file_path).replace('.json', f'_{args.division_type}.json')
    cfg['dataset']['json_file'] = os.path.join(json_file_dir, json_file_name)

    backbone = args.backbone
    division_type = args.division_type
    output_folder_name = f"{backbone}_{division_type}"
    if backbone == 'omnivore':
        seg_size = int(cfg['dataset']['num_frames'] / cfg['dataset']['default_fps'])
        reg_range = len(cfg['model']['regression_range'])
        output_folder_name += f"_{seg_size}s"
        if 'sub' in cfg['dataset']['feat_folder']:
            feat_folder = cfg['dataset']['feat_folder']
            sub_sample_size_str = os.path.basename(feat_folder).split('_')[-1]
            output_folder_name += f"_{sub_sample_size_str}"
    elif backbone == 'videomae':
        cfg['dataset']['input_dim'] = 400
        cfg['model']['input_dim'] = 400
    elif backbone == '3dresnet':
        cfg['dataset']['input_dim'] = 400
        cfg['model']['input_dim'] = 400
    elif backbone == 'slowfast':
        cfg['dataset']['input_dim'] = 400
        cfg['model']['input_dim'] = 400
    elif backbone == 'x3d':
        cfg['dataset']['input_dim'] = (192, 64)
        cfg['model']['input_dim'] = (192, 64)
    pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(cfg['output_folder'], cfg['dataset_name'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg['dataset_name'],
            output_folder_name + '_' + args.output
        )
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg['loader'])

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,map_location=lambda storage, loc: storage.cuda(cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\n Start training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        # save ckpt once in a while
        if (
                ((epoch + 1) == max_epochs) or
                ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'state_dict_ema': model_ema.module.state_dict()
            }

            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

    tb_writer.close()
    print("All done!")
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    # Added to CLI
    parser.add_argument('--backbone', default='omnivore', type=str,
                        choices=['omnivore', '3dresnet', 'videomae', 'slowfast', 'x3d'])
    parser.add_argument('--division_type', default='recordings', type=str,
                        choices=['recordings', 'person', 'environment', 'recipes'])
    parser.add_argument('--feat_folder', default='features', type=str, )

    # Default is 30 for all backbones
    parser.add_argument('--num_frames', default=30, type=int, )
    parser.add_argument('--stride', default=30, type=int, )
    args = parser.parse_args()
    main(args)
