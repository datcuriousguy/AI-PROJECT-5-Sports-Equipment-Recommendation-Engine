
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


conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )
populate_browsing_history(conn=conn)

"""
On running it: seeing the lists of product and user ids

user ids


[82, 44, 109, 53, 110, 71, 150, 94, 10, 78, 49, 126, 87, 100, 21, 18, 69, 66, 99, 62, 63, 143, 31, 147, 148, 37, 130, 119, 114, 113, 120, 106, 55, 137, 6, 28, 125, 85, 102, 132, 93, 16, 46, 89, 51, 67, 133, 32, 70, 14, 142, 80, 3, 36, 144, 57, 105, 97, 11, 73, 134, 26, 98, 88, 41, 129, 96, 42, 75, 2, 27, 48, 135, 76, 101, 19, 81, 61, 127, 50, 58, 112, 83, 115, 47, 72, 86, 30, 15, 9, 117, 13, 103, 33, 7, 4, 139, 17, 64, 59, 40, 118, 77, 43, 35, 22, 39, 74, 140, 92, 124, 104, 116, 111, 95, 54, 1, 65, 107, 141, 138, 136, 38, 123, 60, 52, 145, 84, 56, 149, 91, 79, 122, 45, 29, 24, 121, 68, 131, 90, 8, 25, 146, 23, 5, 34, 20, 12, 128, 108]


product ids


[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134]

Process finished with exit code 0

"""
