with open('train.py', 'r') as f:
    code = f.read()

old1 = "model_dir = 'model_arbIn_{}_S{}_pr{}_ps{}_lr{}_step{}x{}'.format(opt.dataset, opt.num_source, opt.psv_range, opt.psv_step, opt.lr, opt.step, opt.reduce)"
new1 = "model_dir = 'model_arbIn_{}_S{}_pr{}_ps{}_lr{}_step{}x{}_shear{}'.format(opt.dataset, opt.num_source, opt.psv_range, opt.psv_step, opt.lr, opt.step, opt.reduce, opt.shear_amount)"

old2 = "model_dir = 'model_fixIn_{}_S{}_pr{}_ps{}_lr{}_step{}x{}'.format(opt.dataset, opt.num_source, opt.psv_range, opt.psv_step, opt.lr,opt.step,opt.reduce)"
new2 = "model_dir = 'model_fixIn_{}_S{}_pr{}_ps{}_lr{}_step{}x{}_shear{}'.format(opt.dataset, opt.num_source, opt.psv_range, opt.psv_step, opt.lr,opt.step,opt.reduce, opt.shear_amount)"

changes = 0
if old1 in code:
    code = code.replace(old1, new1)
    changes += 1
if old2 in code:
    code = code.replace(old2, new2)
    changes += 1

if changes == 0:
    print("✗ Could not find model_dir patterns")
else:
    with open('train.py', 'w') as f:
        f.write(code)
    print(f"✓ Fixed {changes} model_dir patterns to include shear value")

import ast
try:
    ast.parse(open('train.py').read())
    print("✓ train.py still parses correctly")
except SyntaxError as e:
    print(f"✗ SYNTAX ERROR: {e}")
