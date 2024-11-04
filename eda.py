import dask.dataframe as dd
import matplotlib.pyplot as plt

# Load the data
train_data = dd.read_parquet("C:/Users/Floater PC/OneDrive - Hult Students/Documents/GitHub/jane-street-real-time-market-data-forecasting/train.parquet")

# Data types and unique counts
print("Data Types:\n", train_data.dtypes)

# Unique value counts for categorical columns
categorical_columns = train_data.select_dtypes(include='category').columns
unique_counts = {col: train_data[col].nunique().compute() for col in categorical_columns}
print("Unique Values for Categorical Columns:\n", unique_counts)

# Calculate missing values and percentage per column
missing_values = train_data.isnull().sum().compute()
missing_percentage = (missing_values / len(train_data)) * 100
print("Missing Values:\n", missing_values[missing_values > 0])
print("Missing Percentage:\n", missing_percentage[missing_percentage > 0])

# Calculate descriptive statistics for numeric columns
numeric_columns = [col for col in train_data.columns if str(train_data[col].dtype).startswith(('float', 'int'))]
print("Numeric Columns:", numeric_columns)

stats = train_data[numeric_columns].describe(percentiles=[0.25, 0.5, 0.75]).compute()
print("Summary Statistics:\n", stats)

# Plot a histogram for a numeric column (sampled)
sampled_data = train_data.sample(frac=0.1).compute()
column_name = 'weight'
sampled_data[column_name].hist(bins=30)
plt.title(f'Distribution of {column_name}')
plt.xlabel(column_name)
plt.ylabel('Frequency')
plt.show()

# Calculate correlation matrix for numeric features
correlation_matrix = sampled_data.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Generate rolling mean for a time-indexed feature
time_column = 'time_id'
column_name = 'responder_6'
if time_column in train_data.columns:
    # Sample a smaller portion of data first
    sample_size = 0.01  # 1% of data
    sampled_data = train_data.sample(frac=sample_size).compute()
    
    # Process the sampled data
    sorted_sample = sampled_data.sort_values(time_column).set_index(time_column)
    sorted_sample['rolling_avg'] = sorted_sample[column_name].rolling(window=5).mean()
    
    # Plot the results
    sorted_sample[[column_name, 'rolling_avg']].plot()
    plt.title(f'Rolling Average of {column_name}')
    plt.xlabel('Time ID')
    plt.ylabel('Value')
    plt.show()
    
category_stats = train_data.groupby('partition_id', observed=True)[numeric_columns].mean().compute()
print("Average Values by Category:\n", category_stats)
