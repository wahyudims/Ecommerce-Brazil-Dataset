# Ecommerce-Brazil-Dataset
Here I tried to do analysis about Brazilian ecommerce public dataset of orders made at Olist Store. The analysis consists of product sold analysis, the best time to advertise product, etc. I also try to cluster the customers using RFM analysis into several categories and make a machine learning model which could predict the category of customer based on the clustering that have been made in RFM Analysis

## Context
This dataset is about Ecommerce Store in Brazil. This dataset consists of 7 sub dataset that contain various features from multiple dimensions: from order status, price, payment and freight performance to customer location, product attributes and finally reviews written by customers.

## Task
Create analysis based on several deep dive question and make a model that could predict the category of the customer

## Dataset Info
The dataset split into 7 sub datasets. The first one is customer_dataset.csv which contains personal information about the customer.
  * 'customer_id' : Customer ID,
  * 'customer_unique_id' : unique identifier of a customer,
  * 'customer_zip_code_prefix' : first five digits of customer zip code,
  * 'customer_city' : customer city name, 
  * 'customer_state' : customer state.
  
Second is geolocation_dataset.csv which contains information about cities and states in Brazil.
  * 'geolocation_zip_code_prefix' : first 5 digits of zip code, 
  * 'geolocation_lat' : latitude, 
  * 'geolocation_lng' : longitude,
  * 'geolocation_city' : city name, 
  * 'geolocation_state : state name'.
  
Third is order_item_dataset.csv which contains information of item purchased
  * 'order_id' : order ID, 
  * 'order_item_id' : sequential number of order, 
  * 'product_id' : product unique identifier, 
  * 'seller_id' : seller ID,
  * 'shipping_limit_date' : shipping limit date for handling the order, 
  * 'price' : item price, 
  * 'freight_value' : freight value.
  
Fourth is order_payments_dataset.csv which contains information of payments for each order.
  * 'order_id' : order ID, 
  * 'payment_sequential' : payment with more than one method, 
  * 'payment_type' : method of payment,
  * 'payment_installments' : number of installment, 
  * 'payment_value' : payment value.
  
Fifth is order_reviews_dataset which contains information of review for each order.
  * 'review_id' : review ID, 
  * 'order_id' : order ID, 
  * 'review_score' : review score from 1-5,
  * 'review_comment_title' : review title,
  * 'review_comment_message' : review message, 
  * 'review_creation_date' : review date sent to customer,
  * 'review_answer_timestamp' : review date answered.
  
Sixth is orders_dataset which contains information of the order itself.
  * 'order_id' : order ID, 
  * 'customer_id' : customer ID, 
  * 'order_status' : order status (delivered, shipped, etc.),
  * 'order_purchase_timestamp' : order purchase timestamp,
  * 'order_approved_at' : order approved date, 
  * 'order_delivered_carrier_date' : order delivered to logistic date,
  * 'order_delivered_customer_date' : order delivered to customer date, 
  * 'order_estimated_delivery_date' : order delivered date.
  
Seventh is products_dataset which contains the product information itself.
  * 'product_id' : product ID, 
  * 'product_category_name' : product name (Portuguese), 
  * 'product_name_lenght' : product name length,
  * 'product_description_lenght' : product description length, 
  * 'product_photos_qty' : Photo quantity of product, 
  * 'product_weight_g' : product weight,
  * 'product_length_cm' : product length, 
  * 'product_height_cm' : product height, 
  * 'product_width_cm' : product width.
  
Eighth is sellers_dataset.csv which contains information about seller who sold the product.
  * 'seller_id' : seller ID,
  * 'seller_zip_code_prefix' : first five digits of seller zip code,
  * 'seller_city' : seller city name,
  * 'seller_state' : seller state name.
  
Last is product_category_name_translation which contains information about product's name in english.
  * 'product_category_name' : product's name (Portuguese), 
  * 'product_category_name_english' : product's name (English).
