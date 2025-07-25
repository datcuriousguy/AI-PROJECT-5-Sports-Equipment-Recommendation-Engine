I believe its important to touch on - in a bit more detail - how this model would work, and why we are doing it. Might as well make it useful.

The model we are designing must be trained on pre-existing / first-time interactions with products. We get user_ids and the *level of interest* they showed in products. This takes the form of:

1. hover time
2. dwell time on a product
3. did they click on the product?

These three (essentially) numbers are indicators of interest in each product, and we can think of it like a table - one in which they are compressed into one number - per user-product mappin

This matrix is expected to be rather empty - most products are not interacted with.

We create what are called *dense vectors* which are essentially decimals (to around 3 decimal places) which represent very precise degrees of *how much a product was ‘liked’ by a user*.

This is created by multiplying the *neural network’s ide*a of a unique user, with the *neural network’s idea* of a unique product. Esssentially;
