with open('train.py', 'r') as f:
    code = f.read()

old = '''            for i, batch in enumerate(train_loader, 1):
                ind_source, input, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                disp, inter_lf, pred_lf = model(ind_source, input, opt)'''

new = '''            for i, batch in enumerate(train_loader, 1):
                ind_source, input, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                # NEW: apply shearing if enabled
                if opt.shear_amount != 0.0:
                    from utils.util import apply_shear
                    input = apply_shear(input, ind_source[0], opt.angular_out, opt.shear_amount)
                disp, inter_lf, pred_lf = model(ind_source, input, opt)'''

if 'apply_shear' in code:
    print("✗ apply_shear call already in train.py, skipping")
elif old not in code:
    print("✗ Could not find expected pattern in training loop")
    print("  Showing what the loop currently looks like:")
    import re
    match = re.search(r'for i, batch in enumerate.*?disp, inter_lf, pred_lf = model.*?\)', code, re.DOTALL)
    if match:
        print(match.group(0))
else:
    code = code.replace(old, new)
    with open('train.py', 'w') as f:
        f.write(code)
    print("✓ Successfully injected shearing into training loop")

import ast
try:
    ast.parse(open('train.py').read())
    print("✓ train.py still parses correctly")
except SyntaxError as e:
    print(f"✗ SYNTAX ERROR introduced: {e}")
