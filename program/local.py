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
# Config to suppress warning
import warnings
warnings.filterwarnings("ignore")
os.environ['SPARK_LOCAL_IP'] = 'your_local_ip'
RUNNING = 'local' # 'online'
DATASET = 'kc_house_data.csv'

# Prepare dataset, paths
if RUNNING == 'local':
    cur_dir = os.getcwd()
    data_path = os.path.join(cur_dir, DATASET)
    output_dir = os.path.join(cur_dir, 'output')
    chart_outputs = [
        os.path.join(output_dir, f'c{i}.png') for i in range(1, 5)
    ]
    question_outputs = [
        os.path.join(output_dir, f'q{i}.csv') for i in range(1, 5)
    ]
elif RUNNING == 'online':
    pass

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
selected_df = df.select('bedrooms', 'bathrooms', 'sqft_living', 'price')

# Group by multiple features and calculate average house price
grouped_df = selected_df.groupBy('bedrooms', 'bathrooms', 'sqft_living').agg(avg('price').alias('avg_price'))

# Calculate 0.01 and 0.99 quantiles for avg_price
THRESHOLD_SMALL_PERC = 0.01
THRESHOLD_LARGE_PERC = 0.99
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
    ax.scatter(all_groups['bedrooms'], all_groups['bathrooms'], all_groups['sqft_living'], 
               c='g', marker='o', s=20, alpha=0.5, label='Other groups')

    # Plot significant data points (red, larger, less transparent)
    ax.scatter(significant_groups['bedrooms'], significant_groups['bathrooms'], significant_groups['sqft_living'], 
               c='r', marker='^', s=40, alpha=1.0, label='Significant groups')

    # Set labels
    ax.set_xlabel('Bedrooms')
    ax.set_ylabel('Bathrooms')
    ax.set_zlabel('Sqft Living')
    ax.set_title('3D Plot of House Features')
    # Display legend
    ax.legend()

    # Show plot
    plt.show()
    # Save the plot
    fig.savefig(chart_outputs[0])

# --------------------------------------- Q2 ---------------------------------------
# Select the required columns
df_selected = df.select("zipcode", "yr_built", "price")

# Step 1: Group by zipcode and yr_built, and calculate the average house price
grouped_df = df_selected.groupBy("zipcode", "yr_built").agg(avg("price").alias("avg_price"))

# Step 2: Use pivot to perform data pivoting to compare house price differences across different construction years
pivot_df = grouped_df.groupBy("zipcode").pivot("yr_built").avg("avg_price")

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
    # Sort by zipcode in ascending order
    pivot_pd_df.sort_values(by='zipcode', inplace=True)

    # Divide the chart into 4 subplots, 2 rows and 2 columns
    num_parts = 4
    num_rows, num_cols = 2, 2
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 10))

    # Calculate the number of zipcodes in each subplot
    num_zipcodes = len(pivot_pd_df)
    zipcodes_per_part = (num_zipcodes + num_parts - 1) // num_parts  # Round up

    # Set zipcode as the index
    pivot_pd_df.set_index('zipcode', inplace=True)

    # Transpose DataFrame for plotting
    pivot_pd_df_transpose = pivot_pd_df.transpose()

    # Get the list of years
    years = [int(i) for i in pivot_pd_df_transpose.index.tolist()]

    # Plot each subplot
    for i in range(num_parts):
        start_idx = i * zipcodes_per_part
        end_idx = min((i + 1) * zipcodes_per_part, num_zipcodes)
        zipcodes = pivot_pd_df.index[start_idx:end_idx]

        ax = axs[i // num_cols, i % num_cols]  # Get the axis of the current subplot
        for zipcode in zipcodes:
            ax.plot(years, pivot_pd_df_transpose[zipcode], label=str(zipcode))

        # Add legend
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')

        # Add title and labels
        ax.set_title(f'House Prices Over yr_built by zipcode (Group {i + 1})')
        ax.set_xlabel('Year Built')
        ax.set_ylabel('Average Sale Price')

        # Rotate x-axis labels for better display
        ax.set_xticklabels(ax.get_xticks(), rotation=45)

        # Show grid
        ax.grid(True)

    # Adjust layout to avoid overlap
    plt.tight_layout()
    
    # Display the chart
    plt.show()

    # Save the chart
    fig.savefig(chart_outputs[1])

# --------------------------------------- Q3 ---------------------------------------
# Data preprocessing
# Filter out records with non-null price
df = df.filter(df.price.isNotNull())
# Fill missing values in yr_renovated
df = df.fillna({'yr_renovated': df.agg({'yr_built': 'max'}).collect()[0][0]})

# Feature engineering
# Create IsRenovated column to indicate whether the house is renovated; 1 for renovated, 0 for not renovated
df = df.withColumn("IsRenovated", when(col("yr_renovated") > col("yr_built"), 1).otherwise(0))

# Grouping and aggregation
# Group by IsRenovated and calculate the average and standard deviation of price
grouped_df = df.groupBy("IsRenovated").agg(
    avg("price").alias("AveragePrice"),
    stddev("price").alias("PriceStdDev")
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
# Filter out records with non-null price
df = df.filter(df.price.isNotNull())

# Handle missing values (choose a filling strategy based on specific needs)
df = df.fillna(0)

# Feature selection
# Calculate the correlation between each numeric feature and price
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int' or t[1] == 'double']
correlations = []
for feature in numeric_features:
    if feature != 'price':
        corr_value = df.stat.corr('price', feature)
        correlations.append((feature, corr_value))

# Convert the correlation results to a DataFrame
correlation_df = spark.createDataFrame(correlations, ["Feature", "Correlation"])

# Calculate the correlation of all features with price and sort by correlation in descending order
correlated_features = correlation_df.orderBy(col("Correlation").desc())

# Export the results to a CSV file
correlated_features.coalesce(1).write.csv(question_outputs[3], header=True, mode='overwrite', sep=',')

# Print the top 5 features with the highest correlation
print("[+] Top 5 highly correlated features with price:")
correlated_features.show(n=5)
