# Data Dictionary

## Introduction

This document describes the important columns present in the Olist Brazilian E-Commerce Dataset. It helps users understand the meaning of each attribute used in the project.

---

# Table 1: Customers

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| customer_id | String | Unique identifier for each customer. |
| customer_unique_id | String | Unique customer identifier across multiple orders. |
| customer_zip_code_prefix | Integer | ZIP code prefix of the customer. |
| customer_city | String | Customer's city. |
| customer_state | String | Customer's state. |

---

# Table 2: Orders

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| order_id | String | Unique order identifier. |
| customer_id | String | Customer who placed the order. |
| order_status | String | Current status of the order (delivered, shipped, canceled, etc.). |
| order_purchase_timestamp | DateTime | Date and time when the order was placed. |
| order_approved_at | DateTime | Date and time when the order was approved. |
| order_delivered_carrier_date | DateTime | Date when the carrier received the order. |
| order_delivered_customer_date | DateTime | Date when the customer received the order. |
| order_estimated_delivery_date | DateTime | Estimated delivery date. |

---

# Table 3: Products

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| product_id | String | Unique product identifier. |
| product_category_name | String | Category of the product. |
| product_name_lenght | Integer | Length of the product name. |
| product_description_lenght | Integer | Length of the product description. |
| product_photos_qty | Integer | Number of product images. |
| product_weight_g | Float | Product weight in grams. |
| product_length_cm | Float | Product length in centimeters. |
| product_height_cm | Float | Product height in centimeters. |
| product_width_cm | Float | Product width in centimeters. |