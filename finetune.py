import os
import torch
import cv2
import argparse
import time

from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import Compose
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, ConcatDataset

import wandb

from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, ResizeTrain, NormalizeImage, PrepareForNet, RandomCrop, MirrorSquarePad, ColorAug, RandomHorizontalFlip

from datasets.dataloader import MSDLoader, Trans10KLoader

from loss import ScaleAndShiftInvariantLoss, GradientLoss, MSELoss

def rescale(x, a = 0.0, b = 1.0):
    return a + (b - a)*((x - x.min())/(x.max() - x.min()))


def run(args):
    """Run MonoDepthNN to train on novel depth maps."""
    
    training_datasets = args.training_datasets
    training_datasets_dir = args.training_datasets_dir
    training_datasets_txt = args.training_datasets_txt
    output_path= os.path.join(args.output_path, args.exp_name)
    model_path=args.model_path
    model_type=args.model_type

    wandb.init(project = f"finetuning-{model_type}",
               name = args.exp_name,
               config = {"epochs" : args.epochs,
                         "batch_size" : args.batch_size,
                         "model_type" : model_type,
                         "model_path": model_path,
                         "training_datasets" : training_datasets,
                         "training_datasets_dir": training_datasets_dir,
                         "training_datasets_txt": training_datasets_txt,
                         })

    # Select device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s." % device)


    #### MODEL
    # Load network.
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
                RandomHorizontalFlip(prob=0.5),
                ResizeTrain(
                    net_w,
                    net_h,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                RandomCrop(net_w, net_h),
                ColorAug(prob=0.5),
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
        transform = Compose(
                [
                    RandomHorizontalFlip(prob=0.5),
                    MirrorSquarePad(),
                    ResizeTrain(
                        net_w,
                        net_h,
                        resize_target=True,
                        keep_aspect_ratio=False,
                        ensure_multiple_of=32,
                        resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    ColorAug(prob=0.5),
                    normalization,
                    PrepareForNet(),
                ]
            )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    reload = torch.load(model_path)
    if "model_state_dict" in reload.keys():
        checkpoint = reload['model_state_dict']
    else:
        checkpoint = reload
    model.load_state_dict(checkpoint)

    optimizer = torch.optim.NAdam(model.parameters(), lr = 1e-7)
    if "optimizer_state_dict" in reload.keys() and args.continue_train:
        optimizer.load_state_dict(reload['optimizer_state_dict'])


    scheduler = ExponentialLR(optimizer, gamma = 0.95)
    if "scheduler" in reload.keys() and args.continue_train:
        scheduler.load_state_dict(checkpoint['scheduler'])

    ss_loss, grad_loss, mse_loss = ScaleAndShiftInvariantLoss(), GradientLoss(), MSELoss()

    # Un-freeze all layers.
    for param in model.parameters():
        param.requires_grad = True # False        

    # wandb.watch(model, log_freq=100)
    model.to(device)

    ### DATASETS
    t_datasets = []
        
    if "trans10k" in training_datasets:
        idx = training_datasets.index("trans10k")
        train_t10k = Trans10KLoader(training_datasets_dir[idx], training_datasets_txt[idx], transform=transform)
        print("Training Samples Trans10K", len(train_t10k))
        t_datasets.append(train_t10k)
    if "msd" in training_datasets:
        idx = training_datasets.index("msd")
        train_msd = MSDLoader(training_datasets_dir[idx], training_datasets_txt[idx], transform=transform)
        print("Training Samples MSD", len(train_msd))
        t_datasets.append(train_msd)

    training_data = ConcatDataset(t_datasets)
    dataloader_train = DataLoader(dataset = training_data, batch_size = args.batch_size, shuffle = True, num_workers=8)

    running_time = 0.0
    train_step = 0
    for e in trange(args.epochs):
        start_time_epoch = time.time()

        ###---------------[Training loop]---------------###
        print(f"Training phase for epoch {e}: ")

        for img, depth, _ in tqdm(dataloader_train):
            if train_step % args.step_save == 0 and train_step != 0:
                # Save checkpoint.
                torch.save({'epoch': e,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'loss': loss,
                            }, output_path + "/{}_{}.pt".format(model_type, train_step))

            model.train(True) # I think it's redundant...

            # Turn to tensor and send to device.
            sample = img.to(device)
            gt = depth.to(device)
            optimizer.zero_grad()
            prediction = model(sample)

            mask_idx = torch.full(size = prediction.shape, fill_value = 1).to(device)
            loss = ss_loss(prediction, gt, mask_idx) + grad_loss(prediction, gt, mask_idx) + mse_loss(prediction, gt, mask_idx)

            if train_step % args.step_log == 0:
                wandb.log({"train/batch-wise-loss" : loss.detach().cpu()})
            if train_step % args.step_log_images == 0:
                vis_rgbs =  torch.nn.functional.interpolate(sample, scale_factor=0.25, mode="bilinear")
                vis_preds = torch.nn.functional.interpolate(prediction.unsqueeze(1), scale_factor=0.25)
                vis_gts = torch.nn.functional.interpolate(gt.unsqueeze(1), scale_factor=0.25) 
                wandb.log({
                           "train/rgb": wandb.Image(make_grid(vis_rgbs, nrow = 4)), 
                           "train/prediction": wandb.Image(make_grid(vis_preds, nrow = 4)), 
                           "train/groundtruth" : wandb.Image(make_grid(vis_gts, nrow = 4))
                        })
            if torch.isnan(loss) or torch.isinf(loss):
                exit()

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                optimizer.step()
            train_step += 1

        scheduler.step()

        epoch_time = (time.time() - start_time_epoch)
        running_time += epoch_time
        print(f'Epoch {e} done in {epoch_time} s.')

       
    # Save checkpoint.
    torch.save({'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': loss,
                }, output_path + "/{}_{}.pt".format(model_type, train_step))

    # Save final ckpt without optimizer and scheduler 
    torch.save({'model_state_dict': model.state_dict()}, output_path + "/{}_final.pt".format(model_type))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', 
        default='midas-ft',
    )

    # Paths
    parser.add_argument('--training_datasets', 
        nargs='+',
        default=['msd', 'trans10k'],
        help='training datasets'
    )

    parser.add_argument('--training_datasets_dir', 
        nargs='+',
        default=['MSD/', 'Trans10K/'],
        help='list of files for each training dataset'
    )

    parser.add_argument('--training_datasets_txt', 
        nargs='+',
        default=['datasets/msd/train.txt', 'datasets/trans10k/train.txt'],
        help='list of files for each training dataset'
    )

    parser.add_argument('-o', '--output_path', 
        default='./experiment_models',
        help='where to save the model'
    )

    # Model specs
    parser.add_argument('-m', '--model_path', 
        default=None,
        help='path to the trained weights of model'
    )

    parser.add_argument('-t', '--model_type', 
        default='dpt_large',
        help='model type: dpt_large, midas_v21'
    )

    # Training params
    parser.add_argument('-e', '--epochs', 
        default=20,
        type=int,
        help='number of epochs'
    )

    parser.add_argument('-bs', '--batch_size', 
        default=8,
        type=int,
        help='batch_size'
    )

    parser.add_argument('--continue_train',
        action="store_true",
        help='load optimizer and scheduler state dict'
    )

    # Logging params
    parser.add_argument('--step_save', 
        default=5000,
        type=int,
        help='number of steps to save the model'
    )
    parser.add_argument('--step_log', 
        default=10,
        type=int,
        help='number of steps to save the model'
    )
    parser.add_argument('--step_log_images',
        default=1000,
        type=int,
        help='number of steps to save the model'
    )

    args = parser.parse_args()
    print(args)
    
    os.makedirs(os.path.join(args.output_path, args.exp_name), exist_ok=True)

    default_models = {
        "midas_v21" : "weights/Base/midas_v21-base.pt",
        "dpt_large" : "weights/Base/dpt_large-base.pt",
    }
   
    if args.model_path is None:
        args.model_path = default_models[args.model_type]

    # Set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Start fine-tuning.
    run(args) 
    