
import torch.nn as nn

"""
Creating a class called RecommenderNN athat uses the torch nn module.
instead of a feed forward function, we use __init__ because of its reusability.
Pytorch tracks which parameters need to be optimized and does so accordingly.
using feedforward function would require rebuilding the layer every time a
prediction is made, which can be avoided.
"""
