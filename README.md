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
