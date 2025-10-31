import os
import sys

# 1. Get the path to the current file (test_diff.py)
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up two levels to get to the 'subprojects' directory
subprojects_dir = os.path.dirname(os.path.dirname(current_file_dir))

# 3. Build the path to the 'wp4' source folder
#    (Assuming kaolin_test.py is in 'src' with your other wp4 files)
wp4_src_path = os.path.join(subprojects_dir, "wp4_aeroshell_optimizer", "src")

# 4. Add this path to sys.path
if wp4_src_path not in sys.path:
    sys.path.append(wp4_src_path)

# --- Now you can import it like it's in the same folder ---
import kaolin_test
from crude_estimator_diff import calculateCdProxyDifferentiable
from kaolin_test import get_diff_mesh

vertices, _ = get_diff_mesh()

print(calculateCdProxyDifferentiable(vertices))
