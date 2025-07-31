# make_hci_sparse_dense.py
import h5py, torch, numpy as np

SRC = 'LFASR-FS-GAF/LFData/test_HCI.h5'
DST = 'LFASR-FS-GAF/LFData/test_HCI_sd.h5'   # ? same size on disk

K_idx = [0, 8, 72, 80]
dense_idx = [r*9+c for r in range(1,8) for c in range(1,8)]  # 7×7 centre

with h5py.File(SRC,'r') as src, h5py.File(DST,'w') as dst:
    lfi = src['/LFI_ycbcr'][:]              # load into RAM once
    N,_,_,H,W = lfi.shape

    lf = torch.tensor(lfi).float() / 255.
    lf = lf.view(N, 81, H, W)         # flatten 9×9

    sparse = lf[:, K_idx]             # (N,K,H,W)
    dense  = lf[:, dense_idx]         # (N,49,H,W)

    dst.create_dataset('sparse', data=sparse.numpy(), compression='gzip')
    dst.create_dataset('dense',  data=dense.numpy(),  compression='gzip')
print('Wrote', DST)
