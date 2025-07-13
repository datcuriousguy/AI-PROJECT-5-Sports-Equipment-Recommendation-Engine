
import torch.nn as nn

"""
Creating a class called RecommenderNN athat uses the torch nn module.
instead of a feed forward function, we use __init__ because of its reusability.
Pytorch tracks which parameters need to be optimized and does so accordingly.
using feedforward function would require rebuilding the layer every time a
prediction is made, which can be avoided.
"""

class RecommenderNN(nn.Module):  # we start with embed_dim as 32
    def __init__(self, num_users, num_products, num_interactions, embed_dim=32):
        super(RecommenderNN, self).__init__()

        # embed dim is the number of users times the size of the vector used to represent each one (32 is a good start.)
        # a higher embed_dim means better ability to learn more complex patterns but the tradeoff is higher computation.
