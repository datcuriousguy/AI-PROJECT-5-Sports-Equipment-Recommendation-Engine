
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
