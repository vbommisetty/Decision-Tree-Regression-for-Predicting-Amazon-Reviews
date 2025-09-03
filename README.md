# Decision-Tree-Regression-for-Predicting-Amazon-Reviews

This repository contains the solution to a two-part programming assignment focused on large-scale data processing and predictive modeling using Apache Spark. The project's goal is to engineer features from a massive Amazon reviews dataset and then train a machine learning model to predict user ratings.

### The solution demonstrates proficiency in the following areas:

- Apache Spark: Utilizing the PySpark DataFrame API for efficient and scalable data manipulation.

- Feature Engineering: Applying a variety of techniques, including `Word2Vec` for text data and `OneHotEncoder` with `PCA` for categorical data.

- Machine Learning: Building and hyperparameter tuning a `DecisionTreeRegressor` model.

- Data Engineering: Performing complex data cleaning, aggregation, and schema flattening on a real-world dataset.

## Project Structure
The core of the project is implemented in `spark_pipeline.py`, a single, well-structured Python script that contains functions for each of the data engineering and machine learning tasks. This script is designed to be executed in a distributed computing environment.

- `main.py`: The main script that orchestrates the execution of the project's tasks.

- `spark_pipeline.py`: The core script containing the data engineering and machine learning solution.

- `utilities.py`: Contains helper functions and constants used throughout the project.

- `log4j spark properties.properties`: Configuration for Spark logging.

## Methodology
### Part 1: Data Engineering
This section focuses on preparing the raw data for modeling. The key steps include:

- Data Ingestion & Aggregation: Combining user review and product metadata, followed by group-by aggregations to summarize product and user information.

- Schema Flattening: Handling complex array and map data types by flattening the schema to make it compatible with machine learning models.

- Self-Joins: Performing multiple self-joins to enrich the data with relevant features from the same dataset.

- Typecasting & Imputation: Ensuring data types are correct and handling missing values to create a clean, robust dataset.

### Part 2: Predictive Modeling
With the data prepared, the second part of the project focuses on model development:

Feature Transformation:

`Word2Vec`: Applied to product review text to convert it into a numerical vector representation.

One-Hot Encoding & PCA: Used on categorical features to transform them into a format suitable for the model, followed by PCA to reduce dimensionality and combat the curse of dimensionality.

Model Training: A `DecisionTreeRegressor` model was trained on the processed data.

Hyperparameter Tuning: The model's performance was optimized by tuning its hyperparameters, such as `maxDepth`, to find the best configuration.

