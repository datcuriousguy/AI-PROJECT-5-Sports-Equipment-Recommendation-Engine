I believe its important to touch on - in a bit more detail - how this model would work, and why we are doing it. Might as well make it useful.

The model we are designing must be trained on pre-existing / first-time interactions with products. We get user_ids and the *level of interest* they showed in products. This takes the form of:

1. hover time
2. dwell time on a product
3. did they click on the product?

We create what are called *dense vectors* which are essentially decimals (to around 3 decimal places) which represent very precise degrees of *how much a product was ‘liked’ by a user*.

These three (essentially) numbers are indicators of interest in each product, and we can think of it like a table - one in which they are compressed into one number - per user-product mapping

This matrix is expected to be rather empty - most products are not interacted with.

This is created by multiplying the *neural network’s ide*a of a unique user, with the *neural network’s idea* of a unique product. 

We then *teach* the neural network these relationships between a product and user-engagement (those three metrics in one compressed/embedded decimal) by training it. This is where the class RecommenderNN comes in.

```jsx
class RecommenderNN(nn.Module):  # we start with embed_dim as 32
    def __init__(self, num_users, num_products, num_interactions, embed_dim=32):
        super(RecommenderNN, self).__init__()

        # embed dim is the number of users times the size of the vector used to represent each one (32 is a good start.)
        # a higher embed_dim means better ability to learn more complex patterns but the tradeoff is higher computation.
        # we use embedding to make vector values more precise for tracking preferences at a more nuanced level.
        # eg: 	[0, 0, 1, 0, 0] becomes [0.15, -0.02, 0.88, ..., 0.04] float mos. the numbers capture user habits, interests, or patterns
        # Basically, Instead of treating user IDs as raw numbers, learn a vector for each user that captures their behavior
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.product_embed = nn.Embedding(num_products, embed_dim)
        self.interaction_embed = nn.Embedding(num_interactions, embed_dim)

        # the layers.
        # fc = fully connected
        # fc1 and fc2 are hidden
        # fc3 is the output layer
        self.fc1 = nn.Linear(embed_dim * 3 + 1, 64)  # 3 embeds + dwell time
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

```

The magic here is that - the model can now learn which users like which products and why - without being explicitly told.

But here is the wild and slightly scary thing. We will never really know how it comes to recommend something - it is a black box.
