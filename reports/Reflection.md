# Reflection Answers

## 1. What did you learn from this lab?

In this lab, I learned how to organize a machine learning project using a proper folder structure and Jupyter Notebook. I learned how to load multiple CSV files using Pandas, inspect datasets, identify missing values and duplicate records, generate summary statistics, and understand the relationships between different tables in a relational dataset.

---

## 2. Which table acts as the central table? Why?

The **Orders** table acts as the central table because it connects multiple datasets. It links customers with order items, payments, and reviews through the `order_id` and `customer_id` fields, making it the main table in the Olist dataset.

---

## 3. Why is data exploration important before building machine learning models?

Data exploration is important because it helps us understand the structure and quality of the dataset. It allows us to identify missing values, duplicate records, incorrect data types, and relationships between tables. This information is essential for cleaning and preparing the data before applying machine learning algorithms.

---

## 4. Which Python libraries were used in this lab?

The following Python libraries were used:

- Pandas – for data loading and analysis
- NumPy – for numerical operations
- Matplotlib – for data visualization
- Seaborn – for statistical visualization
- Pathlib – for managing file paths

---

## 5. What challenges did you face during this lab?

Initially, I faced difficulty because the Olist dataset had not been downloaded, so the notebook could not locate any CSV files. I also learned how to configure the Python environment and Jupyter Notebook correctly. After downloading the dataset and placing the CSV files in the `data/raw` folder, I was able to complete the analysis successfully.

---

## 6. How will this lab help you in future machine learning projects?

This lab provided a strong foundation for future machine learning projects. It taught me how to organize project files, understand relational datasets, perform exploratory data analysis (EDA), and document the project properly. These are essential steps before data preprocessing, feature engineering, and model building.