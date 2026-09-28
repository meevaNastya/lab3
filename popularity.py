from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("PopularityModel") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

prior = spark.read.csv(
    "data/order_products__prior.csv",
    header=True,
    inferSchema=True
)

products = spark.read.csv(
    "data/products.csv",
    header=True,
    inferSchema=True
)

popularity = prior.groupBy("product_id") \
    .count() \
    .orderBy(F.desc("count"))

recommendations = popularity.join(
    products,
    "product_id"
).select(
    "product_id",
    "product_name",
    "count"
)

print("Топ-10 рекомендаций модели Popularity:")
recommendations.show(10, truncate=False)

recommendations.limit(100).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("results/popularity")

spark.stop()
