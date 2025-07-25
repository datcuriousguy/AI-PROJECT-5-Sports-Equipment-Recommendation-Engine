# Planning the Recommendation Engine
I believe its important to touch on - in a bit more detail - how this model would work, and why we are doing it. Might as well make it useful.
The model we are designing must:

1. Be trained on pre-existing / first-time interactions with products. We get user_ids and the *level of interest* they showed in products. This takes the form of:
    1. hover time
    2. dwell time on a product
    3. did they click on the product?
    
    These three (essentially) numbers are indicators of interest in each product, and we can think of it like a table. 
    
    This matrix is expected to be rather empty - most products are not interacted with.
