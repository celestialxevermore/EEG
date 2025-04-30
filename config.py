import os
import argparse
import datetime

from utils import make_save_path, createFolder, convert_list, str2list_int, str2list, str2dict, print_info, read_json, band_list

def arg():
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()

    # Time
    parser.add_argument('--date', default=now.strftime('%Y-%m-%d'))
    parser.add_argument('--time', default=now.strftime('%H:%M:%S'))

    # Mode
    parser.add_argument('--mode', required=True, choices=["pretrain", "finetune"])

    # Net
    parser.add_argument('--net', required=True)

    # Data
    parser.add_argument('--dataset', default='bcic4_2a')
    parser.add_argument('--train_cont_path', type=str, help="(Optional) Train continue checkpoint path")

    parser.add_argument('--source_subjects', type=str2list_int, default=None)
    parser.add_argument('--target_subject', type=int, required=True)
    parser.add_argument('--labels', default='0,1,2,3', type=str2list_int)
    parser.add_argument('--n_bands', type=int, default=1)
    parser.add_argument('--band', type=band_list, default=[[0, 42]])
    parser.add_argument('--chans', default='all', type=str2list)

    # Special Options
    parser.add_argument('--use_mutual_learning', action="store_true")
    parser.add_argument('--use_multi_source_align', action="store_true")
    parser.add_argument('--use_kl_alignment', action='store_true')
    parser.add_argument('--use_domain_classifier',action='store_true')
    parser.add_argument('--freq_loss_weight', type=float, default=0.5)
    parser.add_argument('--domain_loss_weight', type=float, default=0.5)
    parser.add_argument('--kl_loss_weight', type = float, default = 0.5)

    # Train
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--criterion', default='CEE')
    parser.add_argument('--opt', default='Adam')
    parser.add_argument('--metrics', type=str2list, default=['loss', 'acc'])
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=2e-3)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=288)
    parser.add_argument('--scheduler', '-sch', default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--eta_min', type=float, default=None)

    # Path
    parser.add_argument('--stamp', required=True)
    parser.add_argument('--signature', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    # Misc
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--print_step', type=int, default=5)
    parser.add_argument('--extractor', default="EEGNet")

    args = parser.parse_args()

    # ========================================
    # [✨ 경로 자동 설정 ✨]
    base_path = f"./result/{args.target_subject}/{args.stamp}"

    if args.mode == "pretrain":
        args.use_pretrained = False
        args.save_path = os.path.join(base_path, "pretrain")
    elif args.mode == "finetune":
        args.use_pretrained = True
        args.save_path = os.path.join(base_path, "finetune")
        if args.pretrained_path is None:
            args.pretrained_path = os.path.join(base_path, "pretrain", "checkpoint", "best_model.tar")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    createFolder(args.save_path)

    # ========================================
    # Print
    print_info(vars(args))
    return args
