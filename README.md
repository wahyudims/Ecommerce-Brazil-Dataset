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
  
## Data Modelling Outline
The following are a few steps in the modeling technique used on this project:
1. Import Modules/Packages.
2. Import Dataset.
3. Data Cleaning.
4. Exploratory Data Analysis (EDA).
5. RFM Analysis.
6. Data Preprocessing.
7. Prediction Modelling.
8. Conclusion & Recommendation.

## Load the dataset
As mentioned above, the dataset consists of 7 sub datasets. Those 7 sub datasets will be merged into one big dataframe. The final dataframe will consists of total rows of 113,193 rows and 40 columns.

## Data Cleaning
Data cleaning is used to identify data that has a NaN value so that it can be processed further using machine learning modeling. The missing values of dataset orders_dataset and product_dataset are relatively small so we could just get rid of it. But the order_reviews dataset has so many missing values that we couldn't just remove it. So for now we will fill the missing values with 'No Reviews'.

## Exploratory Data Analysis (EDA)
Before we jump into the EDA. First we will set several deep dive question.
 * What state does give the most customer?
 * What product is the most sold?
 * What is the average spending of each state?
 * What is the most sold product on each state?
 * What times, days, and month are the orders most occurred?
 * What is the average time of delivery?

Let's jump to each deep dive question.
 
* What state does give the most customer? <br><br>
<img src="https://user-images.githubusercontent.com/89758536/223034523-367f9844-4265-4051-b96d-d91b37b60fec.png" width="1000"> <br>
  According to pareto rule. The object that falls below 80% cut off line are 'vital few' factors that affect selling the most. As we can see the state that gives the most customer are:
  1. SP (Sao Paulo),
  2. RJ (Rio de Janeiro),
  3. MG (Minas Gerais),
  4. RS (Rio Grande do Sul),
  5. PR (Paraná), 
  6. SC (Santa Catarina). 

  Those states alone contributes to 80% of the customer.
  
* What product is the most sold? <br><br>
![image](https://user-images.githubusercontent.com/89758536/223039596-de1a7726-a170-426b-933e-c6ff3bbb1d02.png)
  7 of 71 products sold already cover over than 50% of the sales. Those 7 products are:
  1. Bed Bath Table, 
  2. Health Beauty, 
  3. Sports Leisure,
  4. Furniture Decor,
  5. Computers Accessories,
  6. Housewares,
  7. Watches_gifts.

* What is the average spending of each state? <br><br>
![image](https://user-images.githubusercontent.com/89758536/223040051-d528e0b9-d8be-4ee6-91fa-c4f35fb3f13b.png)
  Paraiba has the highest average spending value, Sao Paulo on the other hand lies on the other end of the same spectrum.

* What is the most sold product on each state?
![image](https://user-images.githubusercontent.com/89758536/223040470-d14296c0-ac04-42a7-8c9e-195249fe8945.png)
  The most sold furnitures across each state are bed bath table, furniture decor, health beauty, and sports leisure. Bed bath table dominate the sales on top 5 states which are SP, RJ, MG, RS, and PR.
  
* What times, days, and month are the orders most occurred?
