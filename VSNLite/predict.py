import argparse
import logging
import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from NetModel import *
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from utils.img_process import *


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', required=True,
                        help="Specify the model to be used")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--is-dir', '-d', action='store_true', default=False,
                        help="Declare the input path as a directory")

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true', default=False,
                        help="Visualize the images as they are processed")
    parser.add_argument('--no-save', '-n', action='store_true', default=False,
                        help="Do not save the output masks")
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help="Minimum probability value to consider a mask pixel white")
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help="Scale factor for the input images")
    parser.add_argument('--get-MCR', action='store_true', default=False,
                        help="Save the maximum connected region rather than the raw predict result")

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def input_dir_solution(args):
    inputFileList = []
    outFileList = []

    # 为输入文件列表添加路径前缀
    inputPath = args.input
    for f in os.listdir(inputPath[0]):
        if os.path.isfile(inputPath[0] + '/' + f):
            inputFileList.append(inputPath[0] + '/' + f)

    # 创建输出文件夹
    if not args.no_save:
        _outFilePath = inputPath[0] + "/out"
        if not os.path.exists(_outFilePath):
            os.mkdir(_outFilePath)

    # 为输出文件列表添加路径前缀以及后缀名
    for f in inputFileList:
        _tempList1 = os.path.splitext(f)
        _tempList2 = os.path.split(_tempList1[0])
        outFileList.append("{}/out/{}{}".format(_tempList2[0], _tempList2[1], _tempList1[1]))

    return inputFileList, outFileList


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = torch.nn.Module()
    if args.model == 'VSNLite4M':
        net = VSNLite4M(n_channels=3, n_classes=1)
    elif args.model == 'VSNLite1MD':
        net = VSNLite1MD(n_channels=3, n_classes=1)
    else:
        logging.error(f'Undefined model {args.model}, trainning terminated')
        sys.exit(0)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model + '.pth', map_location=device))

    logging.info("Model loaded !")

    # True 表示输入为文件夹路径，False 则输入为文件名
    if args.is_dir:
        in_files, out_files = input_dir_solution(args)

    _totalTime = 0

    for i, fn in enumerate(in_files):

        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        _beginTime = time.perf_counter()

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        _predictTime = time.perf_counter() - _beginTime
        _totalTime += _predictTime
        print("Predict using time: ", _predictTime, "Ave: ", _totalTime / (i + 1))

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            if args.get_MCR:
                _tempMat = conv_img2cvmat(result)
                _tempMat, _ = get_largest_area(_tempMat)
                result = conv_cvmat2img(_tempMat)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
