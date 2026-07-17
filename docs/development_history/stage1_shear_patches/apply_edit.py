# apply_edit.py — adds shearing to test_pretrained.py's predict_y
import re

with open('test_pretrained.py', 'r') as f:
    code = f.read()

# Find the predict_y function and inject shearing at the start
old = """def predict_y(model, input, opt):
    # coarse view synthesis
    inter_lf = model.net_view(torch.from_numpy(opt.input_ind), input, opt)"""

new = """def predict_y(model, input, opt):
    # NEW: pre-shear inputs if shear_amount is set
    import torch
    shear_amount = getattr(opt, 'shear_amount', 0.0)
    if shear_amount != 0.0:
        from utils.util import apply_shear
        input = apply_shear(input, torch.from_numpy(opt.input_ind), opt.angular_out, shear_amount)
        print(f">>> shear applied: amount={shear_amount}, input mean after shear={input.mean().item():.6f}")
    # coarse view synthesis
    inter_lf = model.net_view(torch.from_numpy(opt.input_ind), input, opt)"""

if old in code:
    code = code.replace(old, new)
    with open('test_pretrained.py', 'w') as f:
        f.write(code)
    print("✓ Successfully patched test_pretrained.py")
else:
    print("✗ Could not find expected code pattern. Showing current predict_y:")
    # Find and show the function
    match = re.search(r'def predict_y\(.*?\n.*?inter_lf = model\.net_view.*?\n', code, re.DOTALL)
    if match:
        print(match.group(0))
        