
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
        # we use embedding to make vector values more precise for tracking preferences at a more nuanced level.
        # eg: 	[0, 0, 1, 0, 0] becomes [0.15, -0.02, 0.88, ..., 0.04] float mos. the numbers capture user habits, interests, or patterns
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.product_embed = nn.Embedding(num_products, embed_dim)
        self.interaction_embed = nn.Embedding(num_interactions, embed_dim)

        # the layers.
        self.fc1 = nn.Linear(embed_dim * 3 + 1, 64)  # 3 embeds + dwell time
        self.fc2 = nn.Linear(64, 32)
