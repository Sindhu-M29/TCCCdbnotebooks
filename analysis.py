spark.conf.set("fs.azure.account.key.tcccdatalake.dfs.core.windows.net", "#storage acc key")

----------------------------------------------------------------------------------------------------------------------
# Define the directory where your JSON files are stored
directory_path = "abfss://tccc-cont@tcccdatalake.dfs.core.windows.net/csv-fie/"

# List all files in the directory
files = dbutils.fs.ls(directory_path)

# Filter for JSON files and sort by modification time
json_files = [file for file in files if file.name.endswith('.json')]
json_files_sorted = sorted(json_files, key=lambda x: x.modificationTime, reverse=True)

# Get the latest JSON file
latest_file = json_files_sorted[0] if json_files_sorted else None

if latest_file:
    print(f"Latest JSON file: {latest_file.path}")

    # Read the latest JSON file into a DataFrame
    df_latest = spark.read.json(latest_file.path)

    # Show the DataFrame
    df_latest.show()

    # Now you can continue processing df_latest as needed
else:
    print("No JSON files found in the directory.")
------------------------------------------------------------------------------------------
# Import necessary libraries
from pyspark.sql.functions import col, lower, regexp_replace, split
from pyspark.ml.feature import StopWordsRemover

# Define the directory where your JSON files are stored
directory_path = "abfss://tccc-cont@tcccdatalake.dfs.core.windows.net/csv-fie/"

# List all files in the directory
files = dbutils.fs.ls(directory_path)

# Filter for JSON files and sort by modification time
json_files = [file for file in files if file.name.endswith('.json')]
json_files_sorted = sorted(json_files, key=lambda x: x.modificationTime, reverse=True)

# Get the latest JSON file
latest_file = json_files_sorted[0] if json_files_sorted else None

if latest_file:
    print(f"Latest JSON file: {latest_file.path}")

    # Read the latest JSON file into a DataFrame
    df = spark.read.json(latest_file.path)

    # Clean and preprocess the text
    df_cleaned = df.withColumn("comment", lower(col("comment")))\
                   .withColumn("comment", regexp_replace(col("comment"), "[^a-zA-Z\\s]", ""))

    # Split the "comment" column into an array of words
    df_cleaned = df_cleaned.withColumn("comment", split(col("comment"), " "))

    # Remove stop words
    remover = StopWordsRemover(inputCol="comment", outputCol="filtered")
    df_cleaned = remover.transform(df_cleaned)

    # Show the cleaned DataFrame
    df_cleaned.show(truncate=False)
else:
    print("No JSON files found in the directory.")
-------------------------------------------------------------------------------------
from pyspark.ml.feature import HashingTF, IDF

# Hashing TF
hashing_tf = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
featurized_data = hashing_tf.transform(df_cleaned)

# IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurized_data)
rescaled_data = idfModel.transform(featurized_data)
---------------------------------------------------------------------------------
rescaled_data.printSchema()
----------------------------------------------------------------------------
from pyspark.sql.functions import when, col, array_contains

# Assuming 'positive_keywords' and 'negative_keywords' are your keyword lists
positive_keywords = ["good", "excellent"]  # Replace with your actual keywords
negative_keywords = ["bad", "terrible"]  # Replace with your actual keywords

# Create the label column based on the presence of keywords
df_cleaned = df_cleaned.withColumn("label", 
    when(array_contains(col("filtered"), positive_keywords[0]), 1)  # Check for the first positive keyword
    .when(array_contains(col("filtered"), negative_keywords[0]), 0)  # Check for the first negative keyword
    .otherwise(None)
)

# You can extend this logic to check for all keywords in the lists if needed
for keyword in positive_keywords[1:]:
    df_cleaned = df_cleaned.withColumn("label", 
        when(array_contains(col("filtered"), keyword), 1).otherwise(col("label"))
    )

for keyword in negative_keywords[1:]:
    df_cleaned = df_cleaned.withColumn("label", 
        when(array_contains(col("filtered"), keyword), 0).otherwise(col("label"))
    )
------------------------------------------------------------------------------------------
df_cleaned.show(truncate=False)
--------------------------------------------------------------------------------------------
from pyspark.ml.feature import CountVectorizer

# Create a CountVectorizer to convert the filtered text into a feature vector
count_vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
model = count_vectorizer.fit(df_cleaned)  # Fit the model to your DataFrame
df_features = model.transform(df_cleaned)  # Transform the DataFrame

# Now, you can check the schema again to see the new 'features' column
df_features.printSchema()
----------------------------------------------------------------------------------
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
----------------------------------------------------------------------------------
from pyspark.ml.feature import CountVectorizer

# Create an instance of CountVectorizer
count_vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")

# Fit the CountVectorizer model on the cleaned DataFrame
cv_model = count_vectorizer.fit(df_cleaned)

# Transform the cleaned DataFrame to create the features column
df_features = cv_model.transform(df_cleaned)

# Show the resulting DataFrame
df_features.select("filtered", "features").show(truncate=False)
-----------------------------------------------------------------------------------
# Check for null values in the label column
null_count = df_features.filter(col("label").isNull()).count()
print(f"Number of null labels: {null_count}")
------------------------------------------------------------------------------
from pyspark.ml.classification import LogisticRegression
------------------------------------------------------------------------
# Create a Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")
--------------------------------------------------------------------------
# Fill null labels with a valid value, e.g., 2
df_features_filled = df_features.fillna({'label': 2})

# Fit the Logistic Regression model with filled data
model = lr.fit(df_features_filled)
--------------------------------------------------------------------------
# Make predictions
predictions = model.transform(df_features_filled)

# Show predictions
predictions.select("id", "comment", "label", "prediction").show(truncate=False)
--------------------------------------------------------------------------------------
# Show the predictions along with true labels
predictions.select("id", "comment", "label", "prediction").show(truncate=False)

# If you want to map predictions to their respective sentiments
predictions_with_sentiment = predictions.withColumn(
    "predicted_sentiment",
    when(col("prediction") == 1, "Positive").otherwise("Negative")
)

# Show predictions with sentiment
predictions_with_sentiment.select("id", "comment", "label", "predicted_sentiment").show(truncate=False)
----------------------------------------------------------------------------------------
