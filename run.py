"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
import numpy as np

from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, ResizeTrain, NormalizeImage, PrepareForNet, RandomCrop, MirrorSquarePad, ColorAug, RandomHorizontalFlip

from utils import parse_dataset_txt

def run(input_path, output_path, dataset_txt, model_path, model_type="large", save_full=False,  mask_path="", cls2mask=[], mean=False, it=5, output_list=False):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large": # DPT-Large
        model = DPTDepthModel(
            path=None,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )
    elif model_type == "midas_v21":
        model = MidasNet(None, non_negative=True)
        net_w, net_h = 384, 384
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # Mirror Square Pad and Resize
        transform = Compose(
                [
                    Resize(
                        net_w,
                        net_h,
                        resize_target=True,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    normalization,
                    PrepareForNet(),
                ]
            )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    checkpoint = torch.load(model_path)

    if 'model_state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    # get input
    dataset_dict = parse_dataset_txt(dataset_txt)
    num_images = len(dataset_dict["basenames"])

    # create output folder
    os.makedirs(output_path, exist_ok=True)
    if output_list:
        fout = open(output_list, "w")

    print("start processing")
    np.random.seed(0)
    for ind, basename in enumerate(dataset_dict["basenames"]):
        img_name = os.path.join(input_path, basename)            
        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input
        img = utils.read_image(img_name)
        if mask_path:
            mask_name = img_name.replace(input_path, mask_path).replace(".jpg",".png")
            mask = cv2.imread(mask_name, 0)

        preds = []
        for _ in range(args.it):
            if mask_path:
                if args.it == 1:
                    color = np.array([0.5, 0.5, 0.5])
                else:
                    color = np.random.random([3])
                for cls in cls2mask:                
                    img[mask == cls] = color

            img_input = transform({"image": img})["image"]
            # compute
            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                prediction = model.forward(sample)

                if save_full:
                    prediction = (
                        torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=img.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                        )
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                else:
                    prediction = prediction.squeeze().cpu().numpy()
                preds.append(prediction)        
        
        prediction = np.median(np.stack(preds,axis=0), axis=0)

        output_dir = os.path.join(output_path, os.path.dirname(basename))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, os.path.splitext(os.path.basename(img_name))[0])

        np.save(filename, prediction.astype(np.float32))
        if output_list:
            fout.write(img_name + " " + filename + ".npy\n")

        utils.write_depth(filename, prediction, bytes=2)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', 
        default='input',
        help='folder with images'
    )

    parser.add_argument('--dataset_txt', 
        default='dataset.txt',
        help='dataset txt file',
    )

    parser.add_argument('--mask_path', 
        default='',
        help='folder with mask images'
    )

    parser.add_argument('--cls2mask', 
        default=[1],
        type=int,
        nargs='+',
        help='classes to mask'
    )

    parser.add_argument('--it', 
        default=1,
        type=int,
        help="number of iteration to run midas"
    )

    parser.add_argument('-o', '--output_path',
        default='output',
        help='folder for output images'
    )

    parser.add_argument('--output_list', 
        default='',
        help='output list of generated depths as txt file'
    )

    parser.add_argument('--save_full_res', 
        action='store_true',
        help='save original resolution'
    )

    parser.add_argument('-m', '--model_weights', 
        default=None,
        help='path to the trained weights of model'
    )

    parser.add_argument('-t', '--model_type', 
        default='dpt_large',
        help='model type: dpt_large, dpt_hybrid, midas_v21_large or midas_v21_small'
    )

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/Base/midas_v21-base.pt",
        "dpt_large": "weights/Base/dpt_large-base.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print(args)
    # compute depth maps
    run(args.input_path, args.output_path, args.dataset_txt, args.model_weights, args.model_type, save_full=args.save_full_res, mask_path=args.mask_path, cls2mask=args.cls2mask, it=args.it, output_list=args.output_list)
