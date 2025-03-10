import os
import os.path as osp
import glob
import time
import torch
from config import get_arguments
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    setup_seed(2024)

    parser = get_arguments()

    args = parser.parse_args()
    args.time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    args.out_dir = 'result/' + args.time

    sample_list = glob.glob('')

    for i, name in enumerate(sample_list):
        print('Training ', name)
        args.input_name = name
        if not os.path.exists(args.input_name):
            print("Image does not exist: {}".format(args.input_name))
            exit()

        args.out_out_dir = os.path.join(args.out_dir, name)
        if not os.path.exists(args.out_out_dir):
            os.makedirs(args.out_out_dir)

        from train_autoencoderkl import *
        from train_ldm import *
        from utils import *

        with open(osp.join(args.out_out_dir, 'parameters.txt'), 'w') as f:
            for o in args.__dict__:
                f.write("{}\t-\t{}\n".format(o, args.__dict__[o]))


        print("Training model ({})".format(args.out_out_dir))
        start = time.time()
        all_real = read_sample(args, args.input_name)
        all_real_dataset = TensorDataset(all_real)

        all_real_loader = DataLoader(all_real_dataset, batch_size=args.aekl_batch_size, shuffle=True)
        train_aekl(args, all_real_loader)
        train_LDM(args, all_real_loader)

        end = time.time()
        elapsed_time = end - start
        print("Time for training: {} seconds".format(elapsed_time))
