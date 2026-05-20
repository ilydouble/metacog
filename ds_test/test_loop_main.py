#  随机性验证。它只做了一件事：测试 numpy.random.permutation 在当前环境下是否能产生一致的结果，这对于实验的复现性（reproduce）很重要
from ds1000_reflexion import run_ds1000_reflexion
import numpy as np

# check if random permutation is deterministic here
r = np.random.permutation(6)
print("Permutation:", r)
