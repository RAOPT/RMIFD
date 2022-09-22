import torch


def BNM(src, tar):
    """ Batch nuclear-norm maximization, CVPR 2020.
    tar: a tensor, softmax target output.
    NOTE: this does not require source domain data.
    """
    # ra opt::when batch si too little, the result could not be calculated, therefore using the transpose!
    tar = tar.transpose(1, 0)

    _, out, _ = torch.svd(tar)
    loss = -torch.mean(out)
    return loss
