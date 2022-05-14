import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dependencies'))
from paddle import fluid
fluid.install_check.run_check()
