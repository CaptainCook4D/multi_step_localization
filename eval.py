# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from actionformer.libs.core import load_config
from actionformer.libs.datasets import make_dataset, make_data_loader
from actionformer.libs.modeling import make_meta_arch
from actionformer.libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


################################################################################
def main(args):
    """0. load config"""
    # ToDo:
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"

    cfg['dataset']['backbone'] = args.backbone
    cfg['dataset']['feat_folder'] = args.feat_folder
    cfg['dataset']['num_frames'] = args.num_frames
    cfg['dataset']['feat_stride'] = args.stride
    cfg['dataset']['division_type'] = args.division_type
    cfg['dataset']['videos_type'] = args.videos_type

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

    print(f"Output folder name: {output_folder_name}")
    # ToDo: override the args.ckpt with the cfg generated ckpt folder
    dataset_name = cfg['dataset_name']
    args.ckpt = os.path.join(cfg['output_folder'], dataset_name, output_folder_name + '_' + str(args.ckpt))

    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch))
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)
    # ckpt_file = './ckpt/omnivore_recordings_4s_reproduce_ptg/epoch_110.pth.tar'

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    if args.backbone == 'omnivore':
        cfg['model']['input_dim'] = 1024
    elif args.backbone == 'videomae':
        cfg['model']['input_dim'] = 400
    elif args.backbone == '3dresnet':
        cfg['model']['input_dim'] = 400
    elif args.backbone == 'slowfast':
        cfg['model']['input_dim'] = 400
    elif args.backbone == 'x3d':
        cfg['model']['input_dim'] = 400

    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')

    # Added to CLI
    parser.add_argument('--backbone', default='omnivore', type=str,
                        choices=['omnivore', '3dresnet', 'videomae', 'slowfast', 'x3d'])
    parser.add_argument('--division_type', default='recordings', type=str,
                        choices=['recordings', 'person', 'environment', 'recipes'])
    parser.add_argument('--feat_folder', default='features', type=str,)

    # Default is 30 for all backbones
    parser.add_argument('--num_frames', default=30, type=int, )
    parser.add_argument('--stride', default=30, type=int,)
    parser.add_argument('--videos_type', default='', type=str,
                        choices=['all', 'normal', 'error'])
    args = parser.parse_args()
    main(args)
