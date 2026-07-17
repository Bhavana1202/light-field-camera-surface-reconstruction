"""Patch test_hybrid.py to use width-strip processing in net_refine (avoids full-image OOM)."""
import ast

path = "test_hybrid.py"
src = open(path).read()

old_block = """            inter_lf = model.net_view(torch.from_numpy(opt.input_ind), input_views, opt)
            pred_y = model.net_refine(inter_lf)
            pred_y = util.crop_boundary(pred_y, opt.test_crop_size).cpu().numpy()"""

new_block = """            inter_lf = model.net_view(torch.from_numpy(opt.input_ind), input_views, opt)

            # width-strip processing of net_refine to avoid OOM at full resolution
            length = 180
            crop = 20
            input_l, input_m, input_r = util.CropPatches_w(inter_lf, length, crop)
            pred_l = model.net_refine(input_l)
            pred_m = torch.Tensor(input_m.shape[0], opt.angular_out * opt.angular_out,
                                  input_m.shape[2], input_m.shape[3]).to(device)
            for i in range(input_m.shape[0]):
                pred_m[i:i + 1] = model.net_refine(input_m[i:i + 1])
            pred_r = model.net_refine(input_r)
            pred_y = util.MergePatches_w(pred_l, pred_m, pred_r,
                                         inter_lf.shape[2], inter_lf.shape[3],
                                         length, crop)
            pred_y = util.crop_boundary(pred_y, opt.test_crop_size).cpu().numpy()"""

assert old_block in src, "anchor block not found -- has test_hybrid.py been edited?"
assert "CropPatches_w" not in src, "already patched"
src = src.replace(old_block, new_block, 1)

ast.parse(src)
open(path, "w").write(src)
print("PATCH OK -- net_refine now uses width-strip inference")
