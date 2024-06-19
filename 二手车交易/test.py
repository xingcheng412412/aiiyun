import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
from scipy.stats import norm
from IPython.display import Image
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

# 使用 inspect 查看 cross_val_score 源代码
source_code = inspect.getsource(cross_val_score)
print(source_code)
