with open('train.py', 'r') as f:
    code = f.read()

old1 = "        model_dir = 'model_arbIn_{}_S{}_pr{}_ps{}_lr{}_step{}x{}'.format(opt.dataset, opt.num_source, opt.psv_range, opt.psv_step, opt.lr, opt.step, opt.reduce)"
new1 = "        model_dir = 'model_arbIn_{}_S{}_pr{}_ps{}_lr{}_step{}x{}_shear{}'.format(opt.dataset, opt.num_source, opt.psv_range, opt.psv_step, opt.lr, opt.step, opt.reduce, opt.shear_amount)"

old2 = "        model_dir = 'model_fixIn_{}_S{}_pr{}_ps{}_lr{}_step{}x{}'.format(opt.dataset, opt.num_source, opt.psv_range, opt.psv_step, opt.lr,opt.step,opt.reduce)"
new2 = "        model_dir = 'model_fixIn_{}_S{}_pr{}_ps{}_lr{}_step{}x{}_shear{}'.format(opt.dataset, opt.num_source, opt.psv_range, opt.psv_step, opt.lr,opt.step,opt.reduce, opt.shear_amount)"

count = 0
if '_shear{}' in code:
    print("model_dir already includes shear — skipping")
else:
    if old1 in code:
        code = code.replace(old1, new1)
        count += 1
    if old2 in code:
        code = code.replace(old2, new2)
        count += 1
    if count == 0:
        print("ERROR: no model_dir patterns matched")
    else:
        with open('train.py', 'w') as f:
            f.write(code)
        print(f"Updated {count} model_dir pattern(s)")

import ast
try:
    ast.parse(open('train.py').read())
    print("train.py still parses correctly")
except SyntaxError as e:
    print("SYNTAX ERROR:", e)
