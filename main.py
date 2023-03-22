import model
import consts
import logging
import os
import argparse
import sys
import torch
from utils import *
from torchvision.datasets.folder import pil_loader
import gc
import torch

logging.basicConfig(level=logging.DEBUG)
torch.autograd.set_detect_anomaly(True)
gc.collect()

# assert sys.version_info >= (3, 6),\
#     "This script requires Python >= 3.6"  # TODO 3.7?

def str_to_bmi_group(s):
    s = str(s).lower()
    if s in ('healthy'):
        return 0
    elif s in ('overweight'):
        return 1
    elif s in ('obese'):
        return 2
    else:
        raise KeyError("No bmi group found")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AgeProgression on PyTorch.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')

    # train params
    parser.add_argument('--epochs', '-e', default=1, type=int)
    parser.add_argument(
        '--models-saving',
        '--ms',
        dest='models_saving',
        choices=('always', 'last', 'tail', 'never'),
        default='always',
        type=str,
        help='Model saving preference.{br}'
             '\talways: Save trained model at the end of every epoch (default){br}'
             '\tUse this option if you have a lot of free memory and you wish to experiment with the progress of your results.{br}'
             '\tlast: Save trained model only at the end of the last epoch{br}'
             '\tUse this option if you don\'t have a lot of free memory and removing large binary files is a costly operation.{br}'
             '\ttail: "Safe-last". Save trained model at the end of every epoch and remove the saved model of the previous epoch{br}'
             '\tUse this option if you don\'t have a lot of free memory and removing large binary files is a cheap operation.{br}'
             '\tnever: Don\'t save trained model{br}'
             '\tUse this option if you only wish to collect statistics and validation results.{br}'
             'All options except \'never\' will also save when interrupted by the user.'.format(br=os.linesep)
    )
    parser.add_argument('--batch-size', '--bs', dest='batch_size', default=64, type=int)
    parser.add_argument('--weight-decay', '--wd', dest='weight_decay', default=1e-5, type=float)
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', default=2e-4, type=float)
    parser.add_argument('--b1', '-b', dest='b1', default=0.5, type=float)
    parser.add_argument('--b2', '-B', dest='b2', default=0.999, type=float)
    parser.add_argument('--shouldplot', '--sp', dest='sp', default=True, type=bool)

    # test params
    parser.add_argument('--age', '-a', required=False, type=int)
    parser.add_argument('--bmi', '-bm', required=False, type=int)
    parser.add_argument('--bmi_Group', '-g', required=False, type=str_to_bmi_group)
    parser.add_argument('--watermark', '-w', action='store_true') #TODO Necessary ????/

    # shared params
    parser.add_argument('--execution',
                        '--ex',
                        dest='execution_mode',
                        choices=('mps', 'cpu', 'cuda'),
                        default='mps',
                        type=str,
                        help='Select:{br}'
                            '\t-mps for MAC OS{br}'
                            '\t-cuda for cuda enabled Nvidia{br}'
                            '\t-cpu for the pain and suffering'.format(br=os.linesep)
                        )
    parser.add_argument('--load', '-l', required=False, default=None, help='Trained models path for pre-training or for testing')
    parser.add_argument('--input', '-i', default=None, help='Training dataset path (default is {}) or testing image path'.format(default_train_results_dir()))
    parser.add_argument('--output', '-o', default='')
    parser.add_argument('-z', dest='z_channels', default=50, type=int, help='Length of Z vector')
    args = parser.parse_args()

    consts.NUM_Z_CHANNELS = args.z_channels
    net = model.Net()

    if args.execution_mode == 'cuda' and torch.cuda.is_available():
        net.cuda()
        print("cuda selected")
    elif args.execution_mode == 'mps':
        net.mps()
        print("mps selected")
    else:
        print("cpu selected")
        net.cpu()

    if args.mode == 'train':
        betas = (args.b1, args.b2) if args.load is None else None
        weight_decay = args.weight_decay if args.load is None else None
        lr = args.learning_rate if args.load is None else None

        if args.load is not None:
            net.load(args.load)
            print("Loading pre-trained models from {}".format(args.load))

        data_src = args.input or consts.DEFAULT_DATA_PATH
        logging.info("Data folder is {}".format(data_src))
        results_dest = args.output or default_train_results_dir()
        os.makedirs(results_dest, exist_ok=True)
        logging.info("Results folder is {}".format(results_dest))

        with open(os.path.join(results_dest, 'session_arguments.txt'), 'w') as info_file:
            info_file.write(' '.join(sys.argv))

        log_path = os.path.join(results_dest, 'log_results.log')
        if os.path.exists(log_path):
            os.remove(log_path)
        logging.basicConfig(filename=log_path, level=logging.DEBUG)


        net.teach(
            dataset_path=data_src,
            batch_size=args.batch_size,
            betas=betas,
            epochs=args.epochs,
            weight_decay=weight_decay,
            lr=lr,
            should_plot=args.sp,
            where_to_save=results_dest,
            models_saving=args.models_saving
        )

    elif args.mode == 'test':
        if args.load is None:
            raise RuntimeError("Must provide path of trained models")

        net.load(path=args.load, slim=True)

        results_dest = args.output or default_test_results_dir()
        if not os.path.isdir(results_dest):
            os.makedirs(results_dest)

        image_tensor = pil_to_model_tensor_transform(pil_loader(args.input)).to(net.device)
        net.test_single(
            image_tensor=image_tensor,
            age=args.age,
            bmi_group=args.bmi_group,
            target=results_dest,
            watermark=args.watermark
        )
