"""
Utils for monoDepth.
"""
import sys
import re
import numpy as np
import cv2

def decode_3_channels(raw, max_depth=1000):
    """Carla format to depth
    Args:
        raw: carla format depth image. Expected in BGR.
        max_depth: max depth used during rendering
    """
    raw = raw.astype(np.float32)
    out = raw[:,:,2] + raw[:,:,1] * 256 + raw[:,:,0]*256*256
    out = out / (256*256*256 - 1) * max_depth
    return out


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)


def read_d(path, scale_factor=256.):
    """Read depth or disp Map
    Args:
        path: path to depth or disp
        scale_factor: scale factor used to decode png 16 bit images
    """

    if path.endswith("pfm"):
        d = read_pfm(path)
    elif path.endswith("npy"):
        d = np.load(path)
    elif path.endswith("exr"):
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        d = d[:,:,0]
    elif path.endswith("png"):
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(d.shape) == 3:
            d = decode_3_channels(d)
        elif d.dtype == np.uint16:
            d = d.astype(np.float32)
            d = d / scale_factor
    else:
        d = cv2.imread(path)[:,:,0]
   
    return d

def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

def write_depth(path, depth, bytes=1):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bytes))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bytes == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bytes == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))


def read_calib_xml(calib_path, factor_baseline=0.001):
    cv_file = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    calib = cv_file.getNode("proj_matL").mat()[:3,:3]
    fx = calib[0,0]
    baseline = float(cv_file.getNode("baselineLR").real())*factor_baseline
    return fx, baseline


def parse_dataset_txt(dataset_txt):
    with open(dataset_txt) as data_txt:
        gt_files = []
        basenames = []
        focals = []
        baselines = []
        calib_files = []

        for line in data_txt:
            values = line.split(" ")

            if len(values) == 2:
                basenames.append(values[0].strip())
                gt_files.append(values[1].strip())

            elif len(values) == 3:
                basenames.append(values[0].strip())
                gt_files.append(values[1].strip())
                calib_files.append(values[2].strip())

            elif len(values) == 4:
                basenames.append(values[0].strip())
                gt_files.append(values[1].strip())
                focals.append(float(values[2].strip()))
                baselines.append(float(values[3].strip()))

            else:
                print("Wrong format dataset txt file")
                exit(-1)
    
    dataset_dict = {}
    if gt_files: dataset_dict["gt_paths"] = gt_files
    if basenames: dataset_dict["basenames"] = basenames
    if calib_files: dataset_dict["calib_paths"] = calib_files
    if focals: dataset_dict["focals"] = focals
    if baselines: dataset_dict["baselines"] = baselines
    return dataset_dict


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