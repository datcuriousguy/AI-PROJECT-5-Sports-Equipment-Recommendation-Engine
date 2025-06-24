
# we need it for manipulating a local mysql db through python
import mysql.connector
# we need json to convert python lists (refer to tags, say, in readme) into strings for uploading to mysql db - i.e., compatibility
import json

def create_and_populate_clients_table():
    # Connect to MySQL...
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )
    # create cursor object
    cursor = conn.cursor()

