# AI-PROJECT-5 | Sports Equipment Recommendations Using Browsing History

### The Situation

We are a SaaS (Software as as Service) firm that offers a subscription-based content-recommendation engine (neural network based) that uses client firms (sports goods sellers)’ customers’ browsing history to recommend products they might like.

### Our Clients

Our clients are sports stores, physical and / or online

### Our Model Solution

- Trained on Customer browsing history
- Used to recommend the new set of products toshow to the browsing customer

---

## Database Design

We will design a mysql database containing the following:

- client:
    - Company Name
    - Product Catalogue
        - Product id
        - Product Name
        - Product Current Price
        - Product Category (multi dimensional)
    - Customer:
        - Name
        - Age
        - email Id
        - Sport Product / Activity Preferences, if any
        - Browsing history (maybe a list??)

<aside>
💡

Note: By Multi-dimensional we mean each product having been designed for a particular set of activities (with a skew towards one or two), ehnce being recommended to customers whose interests in sports or browsing history pertain to the same.

</aside>

### Table: Clients

| Column Name | Data Type | Description |
| --- | --- | --- |
| `client_id` | INT (PK) | Unique ID for each client |
| `company_name` | VARCHAR(100) | Sports brand name |
| `product_catalogue` | TEXT / JSON | Optional: list of product IDs |

### Table: Products

| Column Name | Data Type | Description |
| --- | --- | --- |
| `product_id` | INT (PK) | Unique ID for each product |
| `client_id` | INT (FK → Clients) | The company selling this product |
| `product_name` | VARCHAR(100) | Name of the product |
| `product_price` | DECIMAL(10,2) | Current price in rupees |
| `product_category` | JSON / TEXT | Can store multi-dimensional tags (e.g., `["running", "men", "shoes"]`) |
| `image_url` | TEXT | Optional: for frontend use |

### Table: Customers

| Column Name | Data Type | Description |
| --- | --- | --- |
| `user_id` | INT (PK) | Unique user ID |
| `name` | VARCHAR(100) | Customer name |
| `age` | INT | Age of customer |
| `email` | VARCHAR(100) | Unique email |
| `preferences` | TEXT / JSON | Optional: e.g. `["tennis", "training shoes"]` |
| `device_type` | ENUM | 'mobile', 'desktop', 'tablet' |

### Table: Browsing History

| Column Name | Data Type | Description |
| --- | --- | --- |
| `interaction_id` | INT (PK) | Unique interaction entry |
| `user_id` | INT (FK → Customers) | Who browsed |
| `product_id` | INT (FK → Products) | Which product was browsed |
| `interaction_type` | ENUM | 'view', 'click', 'add_to_cart', 'purchase' |
| `timestamp` | DATETIME | When the interaction happened |
| `session_id` | VARCHAR(100) | To track behavior in sessions |
| `dwell_time_seconds` | INT | Time spent viewing the product (optional, derived) |
