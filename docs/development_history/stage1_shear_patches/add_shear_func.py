shear_function = '''


def apply_shear(img_source, ind_source, an, shear_amount):
    """Pre-shear input SAIs by shifting each one toward the central reference view."""
    import torch
    import torch.nn.functional as functional

    N, num_source, h, w = img_source.shape
    device = img_source.device
    dtype = img_source.dtype

    ind_center = (an - 1) / 2.0

    XX = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, w).expand(N, h, w)
    YY = torch.arange(0, h, device=device, dtype=dtype).view(1, h, 1).expand(N, h, w)

    sheared = torch.zeros_like(img_source)

    for k_s in range(num_source):
        ind_s = ind_source[k_s].type_as(img_source)
        ind_s_h = torch.floor(ind_s / an)
        ind_s_w = ind_s % an

        delta_u = ind_s_w - ind_center
        delta_v = ind_s_h - ind_center

        grid_w = XX + shear_amount * delta_u
        grid_h = YY + shear_amount * delta_v

        grid_w_norm = 2.0 * grid_w / (w - 1) - 1.0
        grid_h_norm = 2.0 * grid_h / (h - 1) - 1.0
        grid = torch.stack((grid_w_norm, grid_h_norm), dim=3)

        view_input = img_source[:, k_s:k_s+1, :, :]
        sheared_view = functional.grid_sample(view_input, grid, align_corners=False)
        sheared[:, k_s, :, :] = sheared_view[:, 0, :, :]

    return sheared
'''

with open('utils/util.py', 'r') as f:
    content = f.read()

if 'def apply_shear' in content:
    print("apply_shear already exists - not adding again")
else:
    with open('utils/util.py', 'a') as f:
        f.write(shear_function)
    print("Successfully appended apply_shear to utils/util.py")

import ast
try:
    ast.parse(open('utils/util.py').read())
    print("utils/util.py still parses correctly")
except SyntaxError as e:
    print("SYNTAX ERROR:", e)
