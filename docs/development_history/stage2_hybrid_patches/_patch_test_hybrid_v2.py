"""Second patch: smaller strips + per-strip memory cleanup, to defeat the remaining OOM."""
import ast

path = "test_hybrid.py"
src = open(path).read()

old_block = """            inter_lf = model.net_view(torch.from_numpy(opt.input_ind), input_views, opt)

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

new_block = """            inter_lf = model.net_view(torch.from_numpy(opt.input_ind), input_views, opt)
            # release Net_view's intermediate PSV memory before refinement
            torch.cuda.empty_cache()

            # width-strip processing with smaller strips + per-strip cleanup
            length = 90
            crop = 20
            input_l, input_m, input_r = util.CropPatches_w(inter_lf, length, crop)

            def _refine_strip(x):
                with torch.no_grad():
                    y = model.net_refine(x).detach()
                torch.cuda.empty_cache()
                return y

            pred_l = _refine_strip(input_l)
            pred_m = torch.empty(input_m.shape[0], opt.angular_out * opt.angular_out,
                                 input_m.shape[2], input_m.shape[3], device=device)
            for i in range(input_m.shape[0]):
                pred_m[i:i + 1] = _refine_strip(input_m[i:i + 1])
            pred_r = _refine_strip(input_r)

            pred_y = util.MergePatches_w(pred_l, pred_m, pred_r,
                                         inter_lf.shape[2], inter_lf.shape[3],
                                         length, crop)
            pred_y = util.crop_boundary(pred_y, opt.test_crop_size).cpu().numpy()

            # free the big intermediates before next scene
            del inter_lf, input_l, input_m, input_r, pred_l, pred_m, pred_r
            torch.cuda.empty_cache()"""

assert old_block in src, "v1 strip block not found -- did patch v1 apply?"
assert "_refine_strip" not in src, "v2 already applied"
src = src.replace(old_block, new_block, 1)
ast.parse(src)
open(path, "w").write(src)
print("PATCH v2 OK -- smaller strips (length=90) + per-strip empty_cache")
