import ast

path = "model/model_lfasr.py"
src = open(path).read()

# --- edit 1: add import after the net_utils import line ---
import_anchor = "from model.net_utils import make_Altlayer, construct_psv_grid, construct_syn_grid"
import_add = import_anchor + "\nfrom model.sadense_refine import SADenseRefine"
assert import_anchor in src, "import anchor not found!"
assert "from model.sadense_refine import SADenseRefine" not in src, "already patched (import)"
src = src.replace(import_anchor, import_add, 1)

# --- edit 2: swap the refinement instantiation in Net_LFASR.__init__ ---
old_inst = "self.net_refine = Net_refine(opt)"
new_inst = "self.net_refine = SADenseRefine(an=opt.angular_out)"
assert old_inst in src, "instantiation anchor not found!"
src = src.replace(old_inst, new_inst, 1)

# verify it still parses
ast.parse(src)
open(path, "w").write(src)
print("PATCH OK")
print("  + import SADenseRefine")
print("  + net_refine now = SADenseRefine(an=opt.angular_out)")
