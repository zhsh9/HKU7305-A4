import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, stddev, corr, desc
from pyspark.sql.window import Window
# Import Azure Blob storage libraries
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
# Config to suppress warning
import warnings
warnings.filterwarnings("ignore")
RUNNING = 'online' # 'online'

# Prepare dataset, paths
if RUNNING == 'local':
    cur_dir = os.getcwd()
    data_path = os.path.join(cur_dir, 'house-prices.csv')
    output_dir = os.path.join(cur_dir, 'output')
    chart_outputs = [
        os.path.join(output_dir, f'c{i}.png') for i in range(1, 5)
    ]
    question_outputs = [
        os.path.join(output_dir, f'q{i}.csv') for i in range(1, 5)
    ]
elif RUNNING == 'online':
    # Azure Blob storage configuration
    AZURE_STORAGE_CONNECTION_STRING = "your_connection_string"
    CONTAINER_NAME = "your_container_name"
    
    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    data_path = f"https://{blob_service_client.account_name}.blob.core.windows.net/{CONTAINER_NAME}/house-prices.csv"
    output_dir = f"https://{blob_service_client.account_name}.blob.core.windows.net/{CONTAINER_NAME}/output"
    chart_outputs = [
        f"{output_dir}/c{i}.png" for i in range(1, 5)
    ]
    question_outputs = [
        f"{output_dir}/q{i}.csv" for i in range(1, 5)
    ]

# Create SparkSession
spark = SparkSession.builder \
    .appName("House Prices EDA") \
    .config("spark.executor.instances", "2") \
    .config("spark.executor.cores", "1") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
print(f"SparkSession created with name 'House Prices EDA', job started time: {time.ctime()}")

# Read the dataset
df = spark.read.csv(
    data_path,
    header=True,
    inferSchema=True
)

# --------------------------------------- Q1 ---------------------------------------
# Select relevant features and target variable
selected_df = df.select('OverallQual', 'GrLivArea', 'GarageCars', 'SalePrice')

# Group by multiple features and calculate average house price
grouped_df = selected_df.groupBy('OverallQual', 'GrLivArea', 'GarageCars').agg(avg('SalePrice').alias('avg_price'))

# Calculate 0.01 and 0.99 quantiles for avg_price
THRESHOLD_SMALL_PERC = 0.10
THRESHOLD_LARGE_PERC = 0.90
RELATIVE_ERROR = 0.0001
quantiles = grouped_df.approxQuantile('avg_price', [THRESHOLD_SMALL_PERC, THRESHOLD_LARGE_PERC], RELATIVE_ERROR)
threshold_small, threshold_large = quantiles

# Filter groups where average price is in the bottom 1% or top 1%
significant_groups_df = grouped_df.filter((col('avg_price') <= threshold_small) | 
                                          (col('avg_price') >= threshold_large))
significant_groups_df.show(n=3)

# Save the results
significant_groups_df.coalesce(1).write.csv(question_outputs[0], header=True, mode='overwrite', sep=',')

if RUNNING == 'local':
    # Convert Spark DataFrames to Pandas DataFrames
    all_groups = grouped_df.toPandas()
    significant_groups = significant_groups_df.toPandas()

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))  # Adjust the figure size
    ax = fig.add_subplot(111, projection='3d')

    # Plot all data points (green, smaller, more transparent)
    ax.scatter(all_groups['OverallQual'], all_groups['GrLivArea'], all_groups['GarageCars'], 
            c='g', marker='o', s=20, alpha=0.5, label='Other groups')
    # Plot significant data points (red, larger, less transparent)
    ax.scatter(significant_groups['OverallQual'], significant_groups['GrLivArea'], significant_groups['GarageCars'], 
            c='r', marker='^', s=40, alpha=1.0, label='Significant groups')

    # Set labels
    ax.set_xlabel('OverallQual')
    ax.set_ylabel('GrLivArea')
    ax.set_zlabel('GarageCars')
    ax.set_title('3D Plot of House Features')
    # Display legend
    ax.legend()

    # Show plot
    plt.show()
    # Save the plot
    fig.savefig(chart_outputs[0])

# --------------------------------------- Q2 ---------------------------------------
# Select the required columns
df_selected = df.select("Neighborhood", "YearBuilt", "SalePrice")

# Step 1: Group by Neighborhood and YearBuilt, and calculate the average house price
grouped_df = df_selected.groupBy("Neighborhood", "YearBuilt").agg(avg("SalePrice").alias("avg_price"))

# Step 2: Use pivot to perform data pivoting to compare house price differences across different construction years
pivot_df = grouped_df.groupBy("Neighborhood").pivot("YearBuilt").avg("avg_price")

# Save the results
pivot_df.coalesce(1).write.csv(question_outputs[1], header=True, mode='overwrite', sep=',')
pivot_df.show(n=3)

# If this is executed in a local environment, the following code with more analysis will be executed
if RUNNING == 'local':
    # Convert Spark DataFrame to Pandas DataFrame
    pivot_pd_df = pivot_df.toPandas()

    # Backfill missing values
    pivot_pd_df = pivot_pd_df.fillna(method='bfill', axis=1)
    pivot_pd_df = pivot_pd_df.fillna(method='ffill', axis=1)

    # Convert Pandas DataFrame back to Spark DataFrame
    filled_pivot_df = spark.createDataFrame(pivot_pd_df)

    # Save the results
    filled_pivot_df.coalesce(1).write.csv(os.path.join(output_dir, f'q2_more.csv'), header=True, mode='overwrite', sep=',')
    filled_pivot_df.show(n=3)

if RUNNING == 'local':
    # Plot the line chart
    fig = plt.figure(figsize=(28, 7))

    # Set the neighborhood as the index
    pivot_pd_df.set_index('Neighborhood', inplace=True)

    # Transpose the DataFrame for plotting
    pivot_pd_df_transpose = pivot_pd_df.transpose()

    # Plot the price curve for each neighborhood
    for neighborhood in pivot_pd_df_transpose.columns:
        plt.plot(pivot_pd_df_transpose.index, pivot_pd_df_transpose[neighborhood], label=neighborhood)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # Add title and labels
    plt.title('House Prices Over Time by Neighborhood')
    plt.xlabel('Year Built')
    plt.ylabel('Average Sale Price')

    # Rotate x-axis labels for better display
    plt.xticks(rotation=45)

    # Show grid
    plt.grid(True)

    # Display the chart
    plt.tight_layout()
    plt.show()
    
    # Save the chart
    fig.savefig(chart_outputs[1])

# --------------------------------------- Q3 ---------------------------------------
# Data preprocessing
# Filter out records with non-null SalePrice
df = df.filter(df.SalePrice.isNotNull())
# Fill missing values in YearRemodAdd
df = df.fillna({'YearRemodAdd': df.agg({'YearBuilt': 'max'}).collect()[0][0]})

# Feature engineering
# Create IsRenovated column to indicate whether the house is renovated; 1 for renovated, 0 for not renovated
df = df.withColumn("IsRenovated", when(col("YearRemodAdd") > col("YearBuilt"), 1).otherwise(0))

# Grouping and aggregation
# Group by IsRenovated and calculate the average and standard deviation of SalePrice
grouped_df = df.groupBy("IsRenovated").agg(
    avg("SalePrice").alias("AveragePrice"),
    stddev("SalePrice").alias("PriceStdDev")
)

# Calculate the price difference before and after renovation
price_difference = grouped_df.collect()
renovated_price = price_difference[0]['AveragePrice']
not_renovated_price = price_difference[1]['AveragePrice']
price_diff = renovated_price - not_renovated_price

# Export the results to a CSV file
grouped_df.coalesce(1).write.csv(question_outputs[2], header=True, mode='overwrite', sep=',')

# Print the results
grouped_df.show()
print(f"Average price for renovated houses: {renovated_price}")
print(f"Average price for not renovated houses: {not_renovated_price}")
print(f"Price difference due to renovation: {price_diff}")

# --------------------------------------- Q4 ---------------------------------------
# Data preprocessing
# Filter out records with non-null SalePrice
df = df.filter(df.SalePrice.isNotNull())

# Handle missing values (choose a filling strategy based on specific needs)
df = df.fillna(0)

# Feature selection
# Calculate the correlation between each numeric feature and SalePrice
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int' or t[1] == 'double']
correlations = []
for feature in numeric_features:
    if feature != 'SalePrice':
        corr_value = df.stat.corr('SalePrice', feature)
        correlations.append((feature, corr_value))

# Convert the correlation results to a DataFrame
correlation_df = spark.createDataFrame(correlations, ["Feature", "Correlation"])

# Calculate the correlation of all features with SalePrice and sort by correlation in descending order
correlated_features = correlation_df.orderBy(col("Correlation").desc())

# Export the results to a CSV file
correlated_features.coalesce(1).write.csv(question_outputs[3], header=True, mode='overwrite', sep=',')

# Print the top 5 features with the highest correlation
print("[+] Top 5 highly correlated features with SalePrice:")
correlated_features.show(n=5)
