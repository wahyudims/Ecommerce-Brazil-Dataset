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
  
Third is order_item_dataset.csv which contains information of item purchased.
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
  
Fifth is order_reviews_dataset.csv which contains information of review for each order.
  * 'review_id' : review ID, 
  * 'order_id' : order ID, 
  * 'review_score' : review score from 1-5,
  * 'review_comment_title' : review title,
  * 'review_comment_message' : review message, 
  * 'review_creation_date' : review date sent to customer,
  * 'review_answer_timestamp' : review date answered.
  
Sixth is orders_dataset.csv which contains information of the order itself.
  * 'order_id' : order ID, 
  * 'customer_id' : customer ID, 
  * 'order_status' : order status (delivered, shipped, etc.),
  * 'order_purchase_timestamp' : order purchase timestamp,
  * 'order_approved_at' : order approved date, 
  * 'order_delivered_carrier_date' : order delivered to logistic date,
  * 'order_delivered_customer_date' : order delivered to customer date, 
  * 'order_estimated_delivery_date' : order delivered date.
  
Seventh is products_dataset.csv which contains the product information itself.
  * 'product_id' : product ID, 
  * 'product_category_name' : product name (Portuguese), 
  * 'product_name_lenght' : product name length,
  * 'product_description_length' : product description length, 
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
  
Last is product_category_name_translation.csv which contains information about product's name in english.
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
  
* What times, days, and month are the orders most occurred?<br><br>
![image](https://user-images.githubusercontent.com/89758536/223061274-d7a6a214-461b-475a-869a-e8730d04a50d.png)<br><br>
![image](https://user-images.githubusercontent.com/89758536/223065863-43ff0490-ec3c-4eb9-a717-f2fd0ffccd1f.png)<br><br>
![image](https://user-images.githubusercontent.com/89758536/223066247-2254b66c-b720-4508-8db5-7d8245f33417.png)<br><br>

There are several times that has the highest order activity record. In terms of hours, the peak hours are around 10 AM - 3 PM and around 8 PM. In terms of day of the week, the order activity is declining from the highest peak which is monday all the way down until saturday which is the lowest activity recorded. In terms of days, There are several peaks which is around 5th, 15th, and 25th of the month.<br>

* What is the average time of delivery?<br><br>
![image](https://user-images.githubusercontent.com/89758536/223069378-583e4903-972c-49aa-9a4e-0f63f2e8ced7.png)<br><br>
From the figure above we could see that The average delivery time is 10 days while the average estimated delivery time is around 20 days. It means the courier had exceeded the expectation of estimated delivery time and could do their job better than expected.<br>

## RFM Analysis
The “RFM” in RFM analysis stands for recency, frequency and monetary value. RFM analysis is a way to use data based on existing customer behavior to predict how a new customer is likely to act in the future. An RFM model is built using three key factors: 

 * Recency value: This refers to the amount of time since a customer’s last interaction with a brand, which can include their last purchase, a visit to a website, use of a mobile app, a “like” on social media and more. Recency is a key metric because customers who have interacted with your brand more recently are more likely to respond to new marketing efforts.

 * Frequency value: This refers to the number of times a customer has made a purchase or otherwise interacted with your brand during a particular period of time. Frequency is a key metric because it shows how deeply a customer is engaged with your brand. Greater frequency indicates a higher degree of customer loyalty.

 * Monetary value: This refers to the total amount a customer has spent purchasing products and services from your brand over a particular period of time. Monetary value is a key metric because the customers who have spent the most in the past are more likely to spend more in the future.

After doing RFM Analysis, we could cluster the customer into 5 main categories which are:
 1. The first one is new_customer which categorize customers who recently made a purchase but the frequency is still quite low.
 2. The second one is inactive which categorize customers who haven't made a purchase in a long time and are considered inactive.
 3. The third one is at_Risk. There are customers who are at risk of becoming inactive or churning.
 4. The fourth one is loyal customer. This category categorize customers who has high recency and high frequency means they are frequently visiting our store and least likely to go to our rival.
 5. The last is champion which means they are our loyal customer who spend most money in our store.<br><br>
 
 Below is the distribution of each category and the means score of each RFM score for each category.<br><br>
 ![image](https://user-images.githubusercontent.com/89758536/223073520-e6a4e058-b4bc-4ca3-867a-8648af157a02.png)<br><br>
 ![image](https://user-images.githubusercontent.com/89758536/223073633-f1b6cdaa-cc67-4cbf-9797-29dc20bea91c.png)

## Data Preprocessing
### Feature Engineering
Not every features will be used in creating machine learning model. I will only keep several features which are: customer_city, review_score, seller_id, price, freight_value, product_category_name, payment_type, and payment_value. I also will add two features which are actual time delivery and estimated time delivery.

### Preprocessing Categorical Variables
There are 4 columns with categorical variables which are customer_city, product_category_name, payment_type, and Segment. customer_city and product_category_name have a lot of unique value which are around 4071 and 71 values. This values is quite big to be handled using one hot encoding so instead we will use frequency encoding. For payment_type we will use one hot encoding, and for Segment we will use ordinal encoding.

## Classification Prediction Modelling
* The dataset split into 80:20 with 80% is training data and 20% is test data
* Because the target is imbalance. Then we can't use accuracy as the metric. We want to minimize false positive and false negative rate because we don't want to incorrectly predict the customer category so we will use recall, precision, and f1-score as our metric.
* The best algorithm is XGBoost which gives us precision, recall, and f1-score of 78%, 73%, and 75%
* It means we can correctly predict customer category therefore we could make a correct approach which we have covered in previous section in RFM analysis based on this model's prediction. For example, <br><br>
 ![image](https://user-images.githubusercontent.com/89758536/223085275-0435efea-a797-4874-bbed-b17c531860c2.png)

## Conclusion and Recommendation
* 80% of the sales contributed by only 6 states of 27 states. Those 6 states are: 
    1. SP (Sao Paulo), 
    2. RJ (Rio de Janeiro), 
    3. MG (Minas Gerais), 
    4. RS (Rio Grande do Sul), 
    5. PR (Paraná),  
    6. SC (Santa Catarina).
    
  The marketing team can focus their effort more in these states.
* Across 71 products, 50% of the sales could be covered by only 7 products. Those 7 products are: 
    1. Bed Bath Table, 
    2. Health Beauty, 
    3. Sports Leisure, 
    4. Furniture Decor, 
    5. Computers Accessories, 
    6. Housewares, 
    7. Watches_gifts.
    
  Based on this insight the company can focus more on producing this type of products and the marketing team can market these products more
* Paraiba has the highest average spending value, Sao Paulo on the other hand lies on the other end of the same spectrum.
* the most sold furnitures across each state are bed bath table, furniture decor, health beauty, and sports leisure. Bed bath table dominate the sales on top 5 states which are SP, RJ, MG, RS, and PR. The company can focus the item's stock of those 5 states with bed bath table.
* There are several times that has the highest order activity record. In terms of hours, the peak hours are around 10 AM - 3 PM and around 8 PM. In terms of day of the week, the order activity is declining from the highest peak which is monday all the way down until saturday which is the lowest activity recorded. In terms of days, There are several peaks which is around 5th, 15th, and 25th of the month. The marketing team can focus their campaign or advertisment on these day/time.
* The actual delivery time mostly faster than the estimated delivery time meaning the logistic partner did a good job.
* Based on RFM Analysis, the customer can be clustered into 5 categories which are:
    * New customer: customers who recently made a purchase but the frequency is still quite low. We can target them with discount/promo to make them engaged with us,
    * Inactive: customers who haven't made a purchase in a long time and are considered inactive, we could send them a survey or personalized email to ask them if they disastified with our service so we could get them back,
    * at Risk: customers who are at risk of becoming inactive or churning. We can treat them the same like inactive customers while also give them some promo to keep them engaged with us.
    * Loyal customer: customers who has high recency and high frequency means they are frequently visiting our store and least likely to go to our rival. We could offer them some personalized recommendation and gives them some kind of membership to encourage them to buy more product so they can level up their membership status.
    * Champion: our loyal customer who spend most money in our store. Treat them with our biggest services!!
    
* The best model to predict customer's category is using XGBoost which could correctly predict 75% of customer category. This model could help us to correctly gives each customer their respective treatment which means we could increase profit and minimize losses. For example, Let's assume the cost to advertise to one customer is 10 dollars. From 1000 customers who become inactive, we can predict around 75 percent of them correctly which means the company could save around 750 dollars of cost. This 750 dollars could be used to focus on other categories that need prioritized more. 
