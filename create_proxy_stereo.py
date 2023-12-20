import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


# This script assumes the same subfolder structure for each root (mono_root, stereo_root, mask_root).

parser = argparse.ArgumentParser()
parser.add_argument('--mono_root', help="folder with mono predictions")
parser.add_argument('--stereo_root', help="folder with stereo predictions")
parser.add_argument('--stereo_ext', default=".npy", help="stereo extension.")
parser.add_argument('--scale_factor_16bit_stereo', default=64, help="16bit scale factor used during saving")
parser.add_argument('--mask_root', default="", help="folder with semantic masks")
parser.add_argument('--output_root', default="results_merge", help="folder with semantic masks")
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()

debug=args.debug
stereo_root=args.stereo_root
mono_root=args.mono_root
mask_root=args.mask_root
output_root=args.output_root
scale_factor_16bit_stereo = args.scale_factor_16bit_stereo
stereo_ext = args.stereo_ext

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, axis=(1, 2))
    a_01 = np.sum(mask * prediction, axis=(1, 2))
    a_11 = np.sum(mask, axis=(1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, axis=(1, 2))
    b_1 = np.sum(mask * target, axis=(1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


for root, dirs, files in os.walk(mono_root):
    for mono_path in files:
        if mono_path.endswith(".npy"):   
            mono_path = os.path.join(root, mono_path)
         
            stereo_path = mono_path.replace(mono_root, stereo_root).replace("camera_00/", "")
            if "npy" in stereo_ext:
                 stereo = np.load(stereo_path)
            elif "png" in stereo_ext:
                stereo_path = stereo_path.replace(".npy", ".png")
                stereo = cv2.imread(stereo_path, -1).astype(np.float32) / scale_factor_16bit_stereo
            
            mono = np.load(os.path.join(mono_root, mono_path))
            mono = cv2.resize(mono, (stereo.shape[1], stereo.shape[0]), cv2.INTER_CUBIC)  

            valid = (stereo > 0).astype(np.float32)
            mono[valid == 0] = 0

            mask_path = mono_path.replace(mono_root, mask_root).replace(".npy", ".png")
            mask = cv2.imread(mask_path, 0)
            mask_transparent = (mask * valid) > 0
            mask_lambertian = ((1 - mask) * valid) > 0

            mono = (mono - np.min(mono[valid > 0])) / (mono[valid > 0].max() - mono[valid > 0].min())
            a, b = compute_scale_and_shift(np.expand_dims(mono, axis=0), np.expand_dims(stereo, axis=0), np.expand_dims(mask_lambertian.astype(np.float32), axis=0))
            mono = mono * a + b

            merged = np.zeros(stereo.shape)
            merged[mask_transparent] = mono[mask_transparent]
            merged[mask_lambertian] = stereo[mask_lambertian]

            output_path = os.path.join(output_root, os.path.dirname(mono_path).replace(mono_root + "/", ""))
            basename = os.path.basename(mono_path)
            os.makedirs(output_path, exist_ok=True)

            if debug:
                plt.subplot(3,2,1)
                plt.title("mask_seg")
                plt.imshow(cv2.resize((mask*255).astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST))
                plt.subplot(3,2,2)
                plt.title("mask_trasp")
                plt.imshow(cv2.resize(mask_transparent.astype(np.float32), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST))
                plt.subplot(3,2,3)
                plt.title("mask_lamb")
                plt.imshow(cv2.resize(mask_lambertian.astype(np.float32), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST))
                plt.subplot(3,2,4)
                plt.title("stereo")
                plt.imshow(cv2.resize(stereo, None, fx=0.25, fy=0.25), vmin=stereo.min(), vmax=stereo.max(), cmap="jet")
                plt.subplot(3,2,5)
                plt.title("mono")
                plt.imshow(cv2.resize(mono, None, fx=0.25, fy=0.25), vmin=stereo.min(), vmax=stereo.max(), cmap="jet")
                plt.subplot(3,2,6)
                plt.title("merged")
                plt.imshow(cv2.resize(merged, None, fx=0.25, fy=0.25), vmin=stereo.min(), vmax=stereo.max(), cmap="jet")
                plt.savefig(os.path.join(output_path, basename.replace(".npy", ".png")))
            else: 
                np.save(os.path.join(output_path, basename), merged)
