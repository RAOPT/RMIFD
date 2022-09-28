import torch, os, CNN_Datasets, networks_transfer, networks_basic, x_tools
import torch.nn as nn

test_object = "basic"

if __name__ == "__main__":
    if test_object == "transfer":
        from train_transfer import get_args, set_random_seed
        args = get_args()
        set_random_seed(args.seed)
        checkpoint_filename = "checkpoint/CQDXGear1D_1-0_cnn1d_adv/checkpoint_max.tar"
        tester = x_tools.test_transfer(args, checkpoint_filename=checkpoint_filename)
        tester.set_up()

    elif test_object == "basic":
        from train_basic import get_args, set_random_seed
        args = get_args()
        set_random_seed(args.seed)
        checkpoint_filename = "checkpoint/CQDXGear1D_cnn1d_basic/checkpoint_max.tar"
        tester = x_tools.test_basic(args, checkpoint_filename=checkpoint_filename)
        tester.set_up()

    print("The Accuracy: {:.4f}, The Test Loss: {:.4f}".format(tester.acc_calculate()[0], tester.acc_calculate()[-1]))
