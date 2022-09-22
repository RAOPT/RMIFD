import configargparse, os, logging, random, math, torch
import numpy as np
from datetime import datetime

from zmq import DEALER
from x_tools.train_util_transfer import train_util_transfer
from x_tools.train_util_transfer_datn import train_util_transfer_datn
from x_tools.train_util_transfer_multi_branch import train_util_transfer_multi_branch


def get_args():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--acc_preview', type=str, default="Best_acc.txt")

    # network related
    parser.add_argument('--num_class', type=int, default=5)
    parser.add_argument('--basic_net', type=str, default='cnn1d')

    # data loading related
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--src_wc', type=str, required=True)
    parser.add_argument('--tgt_wc', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default=
                        # "../../Dataset/CQDX/CQDXBear/ZouBearingFaultRawConstantLoad.mat")
                        "../../Dataset/CQDX/CQDXGear/gear_fault_rawdata_1024000x1_simple.mat")
    parser.add_argument('--normalize_type', type=str, choices=['0-1', '1-1', 'mean-std'], default="0-1",#default="mean-std", #default='0-1',
                        help='data normalization methods')

    # training related
    parser.add_argument('--is_save_checkpoint', type=str, default=False)
    parser.add_argument('--is_continue_train', type=str, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler_type', type=str, default="fix")
    parser.add_argument('--steps', type=str, default="10")

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=8.)
    parser.add_argument('--transfer_loss', type=str, default='adv')
    parser.add_argument('--is_descend', type=int, default=0)
    return parser.parse_args()


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = get_args()
    print(args)
    set_random_seed(args.seed)
    # ######################train_transfer_util######################
    trainer = train_util_transfer(args)
    trainer.set_up()
    trainer.train()
    # ######################train_transfer_util_datn###################
    # trainer = train_util_transfer_datn(args)
    # trainer.set_up()
    # trainer.train_source_net()
    # trainer.train()
    # ######################train_transfer_multi_branch#######################
    # trainer = train_util_transfer_multi_branch(args)
    # trainer.set_up()
    # trainer.train()
