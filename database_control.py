
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

    create_table_query = """
    CREATE TABLE IF NOT EXISTS Clients (
        client_id INT AUTO_INCREMENT PRIMARY KEY,
        company_name VARCHAR(100) NOT NULL,
        product_catalogue TEXT
    );
    """
    cursor.execute(create_table_query)

    # Prepare list of 50 sports companies
    sports_companies = [
        "Nike", "Adidas", "Puma", "Under Armour", "Reebok", "Asics", "New Balance", "Fila",
        "Skechers", "Mizuno", "Columbia Sportswear", "The North Face", "Salomon", "Wilson Sporting Goods",
        "Yonex", "Decathlon", "Lululemon Athletica", "Hoka One One", "Oakley", "Spalding",
        "Brooks Running", "Champion", "Kappa", "Diadora", "Umbro", "Speedo", "Babolat", "Everlast",
        "Callaway Golf", "Titleist", "Rawlings", "Slazenger", "HEAD", "Easton", "Butterfly",
        "Joma", "Prince Sports", "Tecnifibre", "Mountain Hardwear", "Altra Running", "Patagonia",
        "Gymshark", "Merrell", "Inov-8", "Trek", "Cannondale", "Raleigh", "Atomic", "Burton"
    ]

    """
    Note: the s in %s represents the string datatype which is substituted by the column names in the brackets, in the respective order L-R.
    """

    # Insert into table
    insert_query = "INSERT INTO Clients (company_name, product_catalogue) VALUES (%s, %s)"

    for name in sports_companies:

        # some dummy ids for products in the product catalogue
        product_ids = [i for i in range(100, 103)]
        # json.dumps() converts python object into a json type string, which is needed
        # json.dumps([101, 102, 103]) becomes "[101, 102, 103]".
        product_catalogue_json = json.dumps(product_ids)
        cursor.execute(insert_query, (name, product_catalogue_json))

    # Commit and close
    """
    commit() is analogous to "saving the file", and 
    close() is analogous to closing it.
    """
    conn.commit()
    cursor.close()
    conn.close()
    print("Clients table created!")

create_and_populate_clients_table()

""" PRODUCTS TABLE """

def create_products_table():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )

    # again, cursor is needed to represent the builtin cursor of the conn credentials connector.
    cursor = conn.cursor()

    # IF NOT EXISTS ensures no duplication
    query = """
    CREATE TABLE IF NOT EXISTS Products (
        product_id INT AUTO_INCREMENT PRIMARY KEY,
        client_id INT,
        product_name VARCHAR(100) NOT NULL,
        product_price DECIMAL(10,2),
        product_category TEXT,  -- JSON-style string of tags
        stock_quantity INT,
        image_url VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (client_id) REFERENCES Clients(client_id) ON DELETE CASCADE
    );
    """
    cursor.execute(query)
    # Like before, we 'commit' or 'save' the data in the table (or whatever changes have been made)
    conn.commit()
    # we close the cursor and connection. we don't need it to be open anymore (this is the opposite of mysql.connector.connect)
    cursor.close()
    conn.close()
    print("Products table created :)")

# run1: got an error (see commit msg)
# updated pip and mysql-connector and it now works :)

create_products_table()

""" CUSTOMERS TABLE """

import mysql.connector

def create_customers_table():
    # Connect to the existing MySQL database
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )
    cursor = conn.cursor()

    # SQL query to create the Customers table
    # You can see that this matches with the one in the readme.
    query = """
    CREATE TABLE IF NOT EXISTS Customers (
        user_id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100),
        age INT,
        email VARCHAR(100) UNIQUE,
        preferences TEXT,
        device_type ENUM('mobile', 'desktop', 'tablet')
    );
    """

    # Execute and commit like before
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()
    print("Customers table created successfully!")

create_customers_table()

""" BROWSING HISTORY TABLE """

import mysql.connector

def create_browsing_history_table():
    # Connect to your MySQL database
    conn = mysql.connector.connect(
        host="localhost",
        user="root",        # MySQL username
        password="password",    # MySQL password
        database="ai_project_5_database"
    )
    cursor = conn.cursor()

    # SQL query to create the Browsing_History table
    query = """
    CREATE TABLE IF NOT EXISTS Browsing_History (
        interaction_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        product_id INT,
        interaction_type ENUM('view', 'click', 'add_to_cart', 'purchase') NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        session_id VARCHAR(100),
        dwell_time_seconds INT,
        FOREIGN KEY (user_id) REFERENCES Customers(user_id) ON DELETE CASCADE,
        FOREIGN KEY (product_id) REFERENCES Products(product_id) ON DELETE CASCADE
    );
    """

    # same like others
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()
    print("Browsing_History table created successfully")

create_browsing_history_table()

"""
========================================================================================================================
DATABASE POPULATION
========================================================================================================================
"""

import mysql.connector
import json
import random


def populate_clients_table():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )
    cursor = conn.cursor()

    # some sports brands
    companies = [
        "Nike", "Adidas", "Puma", "Under Armour", "Reebok", "Asics", "New Balance", "Fila",
        "Skechers", "Mizuno", "Columbia", "The North Face", "Salomon", "Wilson", "Yonex",
        "Decathlon", "Lululemon", "Hoka", "Oakley", "Spalding"
    ]

    insert_query = "INSERT INTO Clients (company_name, product_catalogue) VALUES (%s, %s)"

    for name in companies:
        # this for loop creates a list of three random product ids per company name
        product_ids = [random.randint(100, 200) for _ in range(3)]
        product_catalogue = json.dumps(product_ids)
        # this list is then turned into a string for being stored in mysql text field catalogue
        cursor.execute(insert_query, (name, product_catalogue))

    conn.commit()
    cursor.close()
    conn.close()
    print("Clients table populated")

#populate_clients_table()
# run1: success

def populate_products_table():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )
    cursor = conn.cursor()

    categories = [["shoes"], ["clothing"], ["equipment"], ["accessories", "outdoor"], ["shoes", "training"]]
    product_names = ["Pro Trainer", "SpeedGrip Shoes", "AllWeather Jacket", "Peak Performance Shorts", "Hydro Bottle",
                     "Wristbands", "Power Racket", "Grip Socks"]

    # getting each client id in order to link it to a product and its related info (product_price, product_category, stock_quantity, image_url)
    cursor.execute("SELECT client_id FROM Clients")
    client_ids = [row[0] for row in cursor.fetchall()]

    # inserting a linked product id to its product's related info
    insert_query = """
    INSERT INTO Products (client_id, product_name, product_price, product_category, stock_quantity, image_url)
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    for _ in range(67):  # realistic number of products, like 67
        # information in each row of the product table
        client_id = random.choice(client_ids)
        name = random.choice(product_names)
        price = round(random.uniform(20, 150), 2)
        category = json.dumps(random.choice(categories))
        stock = random.randint(10, 200)
        img_url = f"https://example.com/images/{name.replace(' ', '_').lower()}.jpg"  # Can be added later once front end work begins maybe
        # executing the insertion query above the for loop
        cursor.execute(insert_query, (client_id, name, price, category, stock, img_url))

    conn.commit()
    cursor.close()
    conn.close()
    print("Products table populated")

import faker

def populate_customers_table():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )
    cursor = conn.cursor()

    # Faker is a module that creates fake data (hopefully!)
    fake = faker.Faker()
    device_types = ['mobile', 'desktop', 'tablet']
    preferences_pool = [["running", "shoes"], ["yoga", "apparel"], ["cycling"], ["tennis"], ["football", "gear"]]
