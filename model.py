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

    # Get valid user_ids so that we can add them to browsing history.
    cursor.execute("SELECT user_id FROM Customers")
    user_ids = [row[0] for row in cursor.fetchall()]

    print('user ids\n\n')
    print(user_ids)
    print('\n')

    # we need product ids to show in the browsing history
    cursor.execute("SELECT product_id FROM Products")
    product_ids = [row[0] for row in cursor.fetchall()]
