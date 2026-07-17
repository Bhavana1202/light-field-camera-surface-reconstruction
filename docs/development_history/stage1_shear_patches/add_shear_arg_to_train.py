with open('train.py', 'r') as f:
    code = f.read()

old = 'parser.add_argument("--angular_in", type=int, default=3, help="angular number of the sparse light field, [AngIn x AngIn](fixed) or AngIn(random)")'

new = '''parser.add_argument("--angular_in", type=int, default=3, help="angular number of the sparse light field, [AngIn x AngIn](fixed) or AngIn(random)")
parser.add_argument("--shear_amount", type=float, default=0.0, help="pre-shear amount in pixels per angular unit (0 = disabled)")'''

if '--shear_amount' in code:
    print("✗ --shear_amount argument already exists, skipping")
elif old not in code:
    print("✗ Could not find expected pattern. Showing argparse section:")
    import re
    matches = re.findall(r'parser\.add_argument.*', code)
    for m in matches[-5:]:
        print(f"  {m}")
else:
    code = code.replace(old, new)
    with open('train.py', 'w') as f:
        f.write(code)
    print("✓ Successfully added --shear_amount argument")

import ast
try:
    ast.parse(open('train.py').read())
    print("✓ train.py still parses correctly")
except SyntaxError as e:
    print(f"✗ SYNTAX ERROR introduced: {e}")
