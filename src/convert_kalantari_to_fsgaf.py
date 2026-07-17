"""Convert SADenseNet Kalantari h5 into FS-GAF format. Run from SADenseNet-main/."""
import h5py
import json
import numpy as np
import argparse
import os

CROP_H, CROP_W = 375, 540


def get_names(f):
    if 'names' in f.attrs:
        return json.loads(f.attrs['names'])
    return list(f.keys())


def center_crop(arr, ch, cw):
    h, w = arr.shape[2], arr.shape[3]
    top = (h - ch) // 2
    left = (w - cw) // 2
    return arr[:, :, top:top+ch, left:left+cw, :]


def convert(src_path, dst_path, mode, verify=True):
    assert mode in ('train', 'test')
    with h5py.File(src_path, 'r') as f:
        names = get_names(f)
        N = len(names)
        print(f"[{mode}] {N} scenes from {src_path}")

        if mode == 'train':
            out = np.zeros((N, 8, 8, CROP_H, CROP_W), dtype=np.float32)
        else:
            out = np.zeros((N, 8, 8, CROP_H, CROP_W, 3), dtype=np.float32)

        for i, name in enumerate(names):
            ycrcb = f[name]['ycrcb'][:]
            ycrcb = center_crop(ycrcb, CROP_H, CROP_W)

            if verify and i == 0 and mode == 'test':
                try:
                    import cv2
                    v = ycrcb[4, 4]
                    u8 = np.clip(v * 255.0, 0, 255).astype(np.uint8)
                    bgr = cv2.cvtColor(u8, cv2.COLOR_YCrCb2BGR)
                    cv2.imwrite('verify_scene0_centerview.png', bgr)
                    print("  wrote verify_scene0_centerview.png (eyeball the colors)")
                except Exception as e:
                    print("  (visual verify skipped:", e, ")")

            if mode == 'train':
                out[i] = (ycrcb[..., 0] * 255.0).astype(np.float32)
            else:
                ycbcr = ycrcb[..., [0, 2, 1]] * 255.0
                out[i] = ycbcr.astype(np.float32)

            if (i+1) % 10 == 0 or i == N-1:
                print(f"  processed {i+1}/{N}")

        key = 'LFI' if mode == 'train' else 'LFI_ycbcr'
        with h5py.File(dst_path, 'w') as g:
            g.create_dataset(key, data=out)
            g.attrs['names'] = json.dumps([str(n) for n in names])
        print(f"[{mode}] wrote {dst_path}  key='{key}'  shape={out.shape}  "
              f"range[{out.min():.1f},{out.max():.1f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_dir', default='data')
    ap.add_argument('--dst_dir', default='data')
    ap.add_argument('--only', choices=['train', 'test', 'both'], default='both')
    args = ap.parse_args()

    if args.only in ('test', 'both'):
        convert(os.path.join(args.src_dir, 'Kalantari_test.h5'),
                os.path.join(args.dst_dir, 'Kalantari_test_fsgaf.h5'), 'test')
    if args.only in ('train', 'both'):
        convert(os.path.join(args.src_dir, 'Kalantari_train.h5'),
                os.path.join(args.dst_dir, 'Kalantari_train_fsgaf.h5'), 'train')
    print("DONE")
