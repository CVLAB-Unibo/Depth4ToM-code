import os
import numpy as np
import skimage.io
import argparse
import cv2
from tqdm import tqdm
from utils import read_d, parse_dataset_txt, compute_scale_and_shift, read_calib_xml
import threading

CATEGORIES = ['All', 'ToM', 'Other']
METRICS = ['delta1.25', 'delta1.20', 'delta1.15', 'delta1.10', 'delta1.05', 'mae', 'absrel', 'rmse']

class evalThread(threading.Thread):
    def __init__(self, idxs, gts, preds, focals, baselines, acc, categories, min_depth=1, max_depth=10000, resize_factor=0.25, baseline_factor=1000, median_scale_and_shift=False):
        super(evalThread, self).__init__()
        self.idxs = idxs
        self.gts = gts
        self.preds = preds
        self.focals = focals
        self.baselines = baselines
        self.min_depth =  min_depth
        self.max_depth = max_depth
        self.acc = acc
        self.categories = categories
        self.baseline_factor = baseline_factor
        self.median_scale_and_shift = median_scale_and_shift
        self.resize_factor = resize_factor
                
    def run(self):
        for idx in self.idxs:
            gt = read_d(self.gts[idx], scale_factor=256.)
            fx = self.focals[idx]
            baseline = self.baselines[idx]
            baseline = baseline * self.baseline_factor

            gt = cv2.resize(gt, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_NEAREST)
            fx = fx * self.resize_factor
            gt = gt.astype(np.float32) * self.resize_factor

            # CLIP DEPTH GT
            gt[gt > fx * baseline / self.min_depth] = 0 # INVALID IF LESS THAN 1mm (very high disparity values)
            gt[gt < fx * baseline / self.max_depth] = 0 # INVALID IF MORE THAN max_depth meters (very small disparity values)

            pred = read_d(self.preds[idx], scale_factor=256.)
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), cv2.INTER_CUBIC)                    
            pred = (pred - np.min(pred[gt > 0])) / (pred[gt > 0].max() - pred[gt > 0].min())
            if self.median_scale_and_shift:
                gt_shifted = gt - gt[gt>0].min()
                scale = np.median(gt_shifted[gt > 0])/np.median(pred[gt > 0])
                pred = pred * scale
                shift = np.median(gt[gt > 0] - pred[gt > 0])
                pred = pred + shift
            else:
                scale, shift = compute_scale_and_shift(np.expand_dims(pred, axis=0), 
                                                       np.expand_dims(gt, axis=0), 
                                                       np.expand_dims((gt > 0).astype(np.float32), axis=0))
                pred = pred * scale + shift
            
            pred = baseline * fx / pred   

            # CLIP PRED TO WORKING RANGE
            pred[np.isinf(pred)] = self.max_depth
            pred[pred > self.max_depth] = self.max_depth
            pred[pred <self. min_depth] = self.min_depth

            gt = baseline * fx / gt
            gt[np.isinf(gt)] = 0

            if len(self.categories) > 1:
                seg_mask = skimage.io.imread(self.gts[idx].replace(os.path.basename(self.gts[idx]), 'mask_cat.png'))
                seg_mask = cv2.resize(seg_mask, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_NEAREST)
                
            for category in self.categories:
                valid = (gt>0).astype(np.float32)

                if category != 'All':
                    if category == "Other":
                        mask0 = seg_mask == 0
                        mask1 = seg_mask == 1
                    else:
                        mask0 = seg_mask == 2
                        mask1 = seg_mask == 3
                    mask = mask0 | mask1
                    mask = mask.astype(np.float32)
                    valid = valid * mask

                if valid.sum() > 0:
                    metrics = booster_metrics(pred, gt, valid)
                    for k in METRICS:
                        self.acc[category][k].append(metrics[k])


# Main evaluation function
def booster_metrics(d, gt, valid):
    error = np.abs(d-gt)
    error[valid==0] = 0

    thresh = np.maximum((d[valid > 0] / gt[valid > 0]), (gt[valid > 0] / d[valid > 0]))
    delta3 = (thresh < 1.25).astype(np.float32).mean()
    delta4 = (thresh < 1.20).astype(np.float32).mean()
    delta5 = (thresh < 1.15).astype(np.float32).mean()
    delta6 = (thresh < 1.10).astype(np.float32).mean()
    delta7 = (thresh < 1.05).astype(np.float32).mean()

    avgerr = error[valid>0].mean()
    abs_rel = (error[valid>0]/gt[valid>0]).mean()

    rms = (d-gt)**2
    rms = np.sqrt( rms[valid>0].mean() )

    return {'delta1.25':delta3*100., 'delta1.20':delta4*100.,'delta1.15':delta5*100., 'delta1.10':delta6*100., 'delta1.05':delta7*100., 'mae':avgerr, 'absrel': abs_rel, 'rmse':rms, 'errormap':error*(valid>0)}


def eval(gts, preds, focals, baselines, min_depth=1, max_depth=10000, resize_factor=0.25, baseline_factor=1000, median_scale_and_shift=False):
    # Check all files OK
    for test_img in preds:
        if not os.path.exists(test_img):
            print("Missing files in the submission")
            exit(-1)

    if not os.path.exists(gts[0].replace(os.path.basename(gts[0]), 'mask_cat.png')):
        categories = ['All']
    else:
        categories = CATEGORIES

    # INIT
    acc = {}
    results = {}
    for category in categories: 
        acc[category] = {}
        results[category] = {}
        for metric in METRICS:
            acc[category][metric] = []
            results[category][metric] = []

    num_samples = len(gts)
    print("Number of samples", num_samples)
    num_workers = 32
    threads = [] 
    for i in range(num_workers):
        start_idx = num_samples//num_workers * i
        if i != num_workers -1:
            end_idx = num_samples//num_workers * (i+1)
        else:
            end_idx = num_samples
        idxs = range(start_idx, end_idx)
        t = evalThread(idxs, gts, preds, focals, baselines, acc, categories, min_depth, max_depth, resize_factor, baseline_factor, median_scale_and_shift)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    
    for category in categories:
        for k in acc[category]:
            results[category][k] = np.array(acc[category][k]).mean()

    return results


def result2string(result):
    result_string = "{:<12}".format("CLASS")
    for k in METRICS:
        result_string += "{:<12}".format(k)
    result_string += "\n"
    for cat in CATEGORIES:
        if cat in result:
            result_string += "{:<12}".format(cat)
            for metric in METRICS:
                tmp = ""
                if metric in result[cat]: tmp = "{:.2f}".format(result[cat][metric])
                result_string += "{:<12}".format(tmp)
            result_string += "\n"
    return result_string


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_root', 
        help='folder with gt'
    )
    parser.add_argument('--pred_root', 
        help='folder with predictions'
    )
    parser.add_argument('--pred_ext', 
        default=".npy",
        help='prediction extension'
    )
    parser.add_argument('--dataset_txt', 
        help='txt file with a set of $basename $gtpath $calib_file or $basename $gtpath $fx $baseline or $basename $gtpath'
    )
    
    parser.add_argument('--output_path', 
        default="results.txt",
        help='output file'
    )
    parser.add_argument('--resize_factor',
        default=0.25, 
        type=float,
        help='resize gt images with this factor. Evaluation will be done at the gt resolution'
    )
    parser.add_argument('--baseline_factor',
        default=1000, 
        type=float,
        help='scale baseline using this factor'
    )
    parser.add_argument('--min_depth',
        default=1, 
        type=float,
        help='min depth in millimeters'
    )
    parser.add_argument('--max_depth',
        default=10000, 
        type=float,
        help='max depth in millimeters'
    )
    parser.add_argument('--median_scale_and_shift',
        action="store_true", 
        help='rescale prediction with median instead of least square scale and shift'
    )
    args = parser.parse_args()

    # Getting dataset paths
    dataset_dict = parse_dataset_txt(args.dataset_txt)

    gt_files = [os.path.join(args.gt_root, f) for f in dataset_dict["gt_paths"]]
    basenames = [os.path.join(args.pred_root, os.path.splitext(f)[0] + args.pred_ext) for f in dataset_dict["basenames"]]

    if "calib_paths" in dataset_dict:
        focals = []
        baselines = []
        for calib_path in dataset_dict["calib_paths"]:
            fx, baseline = read_calib_xml(os.path.join(args.gt_root, calib_path))
            focals.append(fx)
            baselines.append(baseline)
    elif "focals" in dataset_dict and "baselines" in dataset_dict:
        focals = dataset_dict["focals"]
        baselines = dataset_dict["baselines"]
    else:
        print("Missing focals and baselines or calib files")
        exit(-1)

    # Evaluation
    results = eval(gt_files, basenames, focals, baselines, args.min_depth, args.max_depth, args.resize_factor, args.baseline_factor, args.median_scale_and_shift)

    # Saving results
    results_str = result2string(results)
    print(results_str)
    with open(args.output_path, "w") as fout:
        fout.write(results_str)