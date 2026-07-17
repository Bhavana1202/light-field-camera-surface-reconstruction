"""
Inference-time shearing sweep for SADenseNet.

Tests SADenseNet's pretrained model with various shear amounts applied
to input SAIs before model prediction. Used to compare with FS-GAF's
inference-time shearing sweep.

Output: tmp/test/test_model_30Scenes_shear{X}.csv
"""
import argparse
import functools
import json
from os import path
from time import time
import numpy as np
from scipy.ndimage import shift as nd_shift
from multiprocessing import Pool
import os
import h5py
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K

from components.datasets import get_dataset
from components.generator import Generator
from components.model import create_model
from components.config import Config
from components.utils import get_dir, path2img_name
from components.utils.evaluate import calc_score_ycrcb_lf, postprocess
from components.utils.render import render_bgr


def apply_shear_numpy(x, a_out=(8, 8), a_in=(2, 2), shear_amount=0.0):
    """
    Pre-shear input SAIs by shifting each one toward the central reference view.

    Mirrors the apply_shear() function used in FS-GAF, but uses scipy for the
    spatial shift rather than torch.nn.functional.grid_sample.

    x shape: (a_in[0], a_in[1], H, W)  -- the input SAIs from get_xy
    shear_amount: pixels to shift per angular unit
    """
    if shear_amount == 0.0:
        return x

    # Find which absolute angular positions in a_out the a_in views occupy.
    # For 8x8 output and 2x2 input: step=(7,7), positions=(0,7)
    step = (np.array(a_out) - np.array(a_in)) // (np.array(a_in) - 1) + 1

    # Center of the full a_out angular grid
    center_y = (a_out[0] - 1) / 2.0
    center_x = (a_out[1] - 1) / 2.0

    sheared = np.zeros_like(x)
    for i in range(a_in[0]):
        for j in range(a_in[1]):
            abs_y = i * step[0]
            abs_x = j * step[1]
            dy = (abs_y - center_y) * shear_amount
            dx = (abs_x - center_x) * shear_amount
            # nd_shift positive value shifts image in positive direction;
            # we want each view to shift toward center, so negate
            sheared[i, j] = nd_shift(x[i, j], shift=(-dy, -dx),
                                     mode='constant', cval=0, order=1)

    return sheared


def test_sample(bgr, ycrcb, name, model_pth, gpuid, config, to_render_bgr, to_render_diff,
                shear_amount=0.0, dataset=''):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    t_config = tf.ConfigProto()
    t_config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=t_config))

    lf = ycrcb[:, :, :, :, 0]

    model = create_model((ycrcb.shape[2], ycrcb.shape[3]), config)
    model.load_weights(model_pth)

    x, _ = Generator.get_xy(lf, config.a_in)

    # ===== APPLY SHEARING HERE =====
    if shear_amount != 0.0:
        x_before_mean = x.mean()
        x = apply_shear_numpy(x, a_out=config.a_out, a_in=config.a_in,
                              shear_amount=shear_amount)
        x_after_mean = x.mean()
        print(f">>> [{name}] shear={shear_amount}: mean before={x_before_mean:.4f}, after={x_after_mean:.4f}")
    # ================================

    x = np.expand_dims(x, 0)

    t = time()
    y2 = model.predict(x, batch_size=1)
    y2 = postprocess(y2[0])
    t = time() - t

    render_diff = path.join(get_dir(config.dir_tmp_test, dataset+".diff", name), "%d.png") if to_render_diff else False
    res = calc_score_ycrcb_lf(y2, ycrcb, bgr, a_in=config.a_in, a_out=config.a_out,
                              render_diff=render_diff)

    if to_render_bgr:
        pth_dir = get_dir(config.dir_tmp_test, dataset, path2img_name(name))
        render_bgr(y2, ycrcb, config, pth_dir=pth_dir)

    return res + (t,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SADenseNet shearing sweep.')
    parser.add_argument('--model', help="Model path.", type=str)
    parser.add_argument('--bgr', help="Render output's BGR version.", action='store_true')
    parser.add_argument('--diff', help="Render diff maps.", action='store_true')
    parser.add_argument('--gpuid', help="ID of the gpu to be used", type=str, default='0')
    parser.add_argument('--mp', help="Number of process", type=int, default=1)
    parser.add_argument('--shear_amount', help="Pre-shear amount in pixels per angular unit",
                        type=float, default=0.0)
    args = parser.parse_args()

    print(f"=== Running SADenseNet inference with shear_amount={args.shear_amount} ===")

    datasets = ['30Scenes']
    for ds in datasets:
        with h5py.File(get_dataset(ds, Config).get_path_test(), 'r') as h5:
            names = json.loads(h5.attrs['names'])
            n_samples = len(names)

            psnr_lst = np.zeros(n_samples)
            ssim_lst = np.zeros(n_samples)
            elapse_lst = np.zeros(n_samples)
            pbar = tqdm(total=n_samples)

            target = functools.partial(test_sample,
                                       model_pth=args.model, gpuid=args.gpuid, config=Config,
                                       to_render_bgr=args.bgr, to_render_diff=args.diff,
                                       shear_amount=args.shear_amount,
                                       dataset=ds)

            for i in range(0, n_samples, args.mp):
                name_lst = names[i:i + args.mp]
                length = len(name_lst)
                bgr_lst = [h5[names[i + j] + '/bgr'][:] for j in range(length)]
                ycrcb_lst = [h5[names[i + j] + '/ycrcb'][:] for j in range(length)]

                if args.mp > 1:
                    with Pool(args.mp) as pool:
                        res_lst = pool.starmap(target, zip(bgr_lst, ycrcb_lst, name_lst))
                else:
                    res_lst = [target(bgr, ycrcb, name) for bgr, ycrcb, name in zip(bgr_lst, ycrcb_lst, name_lst)]

                psnr_lst[i:i + length] = [res[0] for res in res_lst]
                ssim_lst[i:i + length] = [res[1] for res in res_lst]
                elapse_lst[i:i + length] = [res[2] for res in res_lst]

                pbar.update(len(bgr_lst))

        psnr_score = np.mean(psnr_lst)
        ssim_score = np.mean(ssim_lst)
        elapse = np.mean(elapse_lst)
        print(f"\n=== Shear={args.shear_amount}: Final PSNR={psnr_score:.2f}, SSIM={ssim_score:.4f}, time={elapse:.2f}s ===")

        # Output CSV name includes shear amount so files don't overwrite
        save_pth = path.join(Config.dir_tmp_test,
                             f'test_model_{ds}_shear{args.shear_amount}.csv')
        with open(save_pth, 'w') as f:
            f.write(','.join(['sample', 'psnr', 'ssim', 'time']) + '\n')
            for i in range(len(psnr_lst)):
                f.write(','.join(
                    [path2img_name(names[i]),
                     "%.2f" % psnr_lst[i],
                     "%.4f" % ssim_lst[i],
                     "%.2f" % elapse_lst[i]]
                ) + '\n')
        print(f"Results saved to {save_pth}")
