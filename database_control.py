
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

# create_and_populate_clients_table()

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

# create_products_table()

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

#create_customers_table()

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

#create_browsing_history_table()

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

# populate_clients_table()
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
    INSERT INTO products (client_id, product_name, product_price, product_category, stock_quantity, image_url)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    if client_ids:
        client_id = random.choice(client_ids)
        print(client_id)
    else:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password",
            database="ai_project_5_database"
        )
        cursor = conn.cursor()

        categories = [["shoes"], ["clothing"], ["equipment"], ["accessories", "outdoor"], ["shoes", "training"]]
        product_names = ["Pro Trainer", "SpeedGrip Shoes", "AllWeather Jacket", "Peak Performance Shorts",
                         "Hydro Bottle",
                         "Wristbands", "Power Racket", "Grip Socks"]

        # getting each client id in order to link it to a product and its related info (product_price, product_category, stock_quantity, image_url)
        cursor.execute("SELECT client_id FROM Clients")
        client_ids = [row[0] for row in cursor.fetchall()]
    for _ in range(67):  # realistic number of products, like 67
        # information in each row of the product table


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
populate_products_table()

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

    # for example, i might be browsing on a mobile and have a preference for cycling.

    insert_query = """
    INSERT INTO Customers (name, age, email, preferences, device_type)
    VALUES (%s, %s, %s, %s, %s)
    """
    # and this is where faker comes in!
    for _ in range(50):  # realistic number of users
        name = fake.name()
        age = random.randint(18, 50)
        email = fake.unique.email()
        preferences = json.dumps(random.choice(preferences_pool))
        device = random.choice(device_types)

        cursor.execute(insert_query, (name, age, email, preferences, device))

    conn.commit()
    cursor.close()
    conn.close()
    print("Customers table is populated!")

# populate_customers_table()

def populate_browsing_history_table():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )
    cursor = conn.cursor()

    # Get user IDs
    cursor.execute("SELECT user_id FROM Customers")
    user_ids = [row[0] for row in cursor.fetchall()]

    # Get product IDs
    cursor.execute("SELECT product_id FROM Products")
    product_ids = [row[0] for row in cursor.fetchall()]

    interaction_types = ['view', 'click', 'add_to_cart', 'purchase']

    insert_query = """
        INSERT INTO browsing_history (user_id, product_id, interaction_type, timestamp, session_id, dwell_time_seconds)
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    from datetime import datetime, timedelta

    for _ in range(200):
        user_id = random.choice(user_ids)

        if product_ids:
            product_id = random.choice(product_ids)
        else:
            product_id = random.randint(100, 103)

        interaction = random.choice(interaction_types)
        time = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        session_id = f"sess_{random.randint(1000, 9999)}"
        dwell = random.randint(5, 120)

        cursor.execute(insert_query, (user_id, product_id, interaction, time, session_id, dwell))

    conn.commit()
    cursor.close()
    conn.close()
    print("Browsing_History table populated!")

#populate_browsing_history_table()

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

    print('product ids\n\n')
    print(product_ids)

    # we'll use these to fabricate data for browsing history
    interaction_types = ['view', 'click', 'purchase']
    inserted = 0 # we keep track of number of rows inserted

    for _ in range(num_entries):
        import datetime
        from datetime import timedelta
        from datetime import time
        import time

        user_id = random.choice(user_ids)
        product_id = random.choice(product_ids)
        dwell_time_seconds = random.randint(3,129)
        # for interaction type, keepingga bias erring towards views and kess towards clicks just like in reality.
        interaction_type = random.choices(interaction_types, weights=[0.6, 0.3, 0.1])[0] # 0 as it is just the name of the interaction we want!

        # sincs this is a browsing HISTORY, we need a number of days before when it ewas reorded:
        days_ago = random.randint(0, 30)
        t = time.localtime
        fmt_time = time.strftime("%Y-%m-%d %H:%M", t)
        #integrating days_ago into the bh: previous to the current day:
        timestamp = time.strftime(fmt_time) - timedelta(days=days_ago) # minus => this was in the past.

        insertion_statement = """
            INSERT INTO browsing_history (user_id, product_id, dwell_time_seconds, interaction_type, timestamp)
            VALUES (%s, %s, %s, %s, %s)"""
        cursor.execute(insertion_statement, (user_id, product_id, dwell_time_seconds, interaction_type, timestamp))

        inserted += 1
    print('inserted')
    conn.commit()

conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="ai_project_5_database"
    )
populate_browsing_history(conn=conn)
# run1: success
# despite running the func, selecting * from bh table in mysql cmc gives empty set.
#diagnosing...

# error:
"""
[537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335]
Traceback (most recent call last):
  File "C:\Users\Daniel\AppData\Roaming\JetBrains\PyCharmCE2024.3\scratches\NN_PROJECT_5\database_control.py", line 429, in <module>
    populate_browsing_history(conn=conn)
  File "C:\Users\Daniel\AppData\Roaming\JetBrains\PyCharmCE2024.3\scratches\NN_PROJECT_5\database_control.py", line 410, in populate_browsing_history
    fmt_time = time.strftime("%Y-%m-%d %H:%M", t)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Tuple or struct_time argument required
"""
