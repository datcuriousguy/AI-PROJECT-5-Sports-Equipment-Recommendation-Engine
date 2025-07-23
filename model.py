
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

# Now we need to use the SQL JOIN query to connect browsing_history to customers,
# and products - in a meaningful way such that get a user_id (hence a user)'s dwell time
# on a specific item such as a specific product or category.

# The point of this is to get an understanding of what a specific user_id 'likes' based on mouse
# hovering patterns across specific parts of the page.

# for example, a value of:
# 0 -> add to cart, hover or view
# 1 -> click or purchase

# just realizing i could have used docstring for all this

"""
Apparently, using a dataframe is the most recommended way to go when soing a
mysql + ml thing? if someone can confirm this that would be great :)
"""

import mysql.connector
import pandas as pd

def load_training_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )


# Understanding the query:
    # We extract all user browsing information using a subquery that aliases browsing_history table as bh,
    # we use the join query to create a one-to-one mapping of browsing history with customer ids cause
    # there is no use in having browsing history if it can't be linked to customer ids.
    # we use LIMIT to limit the number of returned records to 500 for now.
    query = """
    SELECT 
        bh.user_id,
        bh.product_id,
        bh.interaction_type,
        bh.dwell_time_seconds,
        bh.session_id,
        bh.timestamp,
        c.preferences
    FROM Browsing_History bh
    JOIN Customers c ON bh.user_id = c.user_id
    JOIN Products p ON bh.product_id = p.product_id
    LIMIT 500;
    """

    # note JOIN always has an ON. tHIS WAS taught to me by mr Sameer Unawane During his time at Jupiter Business Systems. Thank you

    df = pd.read_sql(query, conn)
    # creating the df vusimng the returned result of query

    conn.close()

    df['label'] = df['interaction_type'].apply(
        lambda x: 1 if x in ['click', 'purchase'] else 0
    )
    # as previously discussed, if the action is click / purchase,
    # then we use 1. else its less valuable hence we use 0.

    # using fillna to full in any seconds of time spent 'hovering'
    df['dwell_time_seconds'] = df['dwell_time_seconds'].fillna(0)

# load_training_data()

#run1: error free!

def preprocessing():

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )

    # Load data
    df = pd.read_sql("SELECT user_id, product_id, dwell_time_seconds, interaction_type FROM Browsing_History", conn)
    print(df)

# preprocessing()

"""
on running preprocessing():

Empty DataFrame
Columns: [user_id, product_id, dwell_time_seconds, interaction_type]
Index: []

 -> browsing history is empty. need to populate. guess i missed it
"""

# populating browsing_history table mysql

import random
from datetime import datetime, timedelta

def populate_browsing_history(conn, num_entries=500):
    cursor = conn.cursor()
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )

    # Get valid user_ids so that we can add them to browsing history.
    cursor.execute("SELECT user_id FROM Customers")
    user_ids = [row[0] for row in cursor.fetchall()]

    print('user ids\n\n')
    print(user_ids)
    print('\n')

    # we need product ids to show in the browsing history
    cursor.execute("SELECT product_id FROM Products")
    product_ids = [row[0] for row in cursor.fetchall()]

    print('product ids\n\n')
    print(product_ids)

    # we'll use these to fabricate data for browsing history
    interaction_types = ['view', 'click', 'purchase']
    inserted = 0 # we keep track of number of rows inserted

    for _ in range(num_entries):

        user_id = random.choice(user_ids)
        product_id = random.choice(product_ids)
        dwell_time_seconds = random.randint(3,129)

# populate_browsing_history(conn=conn)
