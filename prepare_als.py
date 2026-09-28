from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("PrepareALS") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

orders = spark.read.csv(
    "data/orders.csv",
    header=True,
    inferSchema=True
)

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

prior_orders = orders \
    .filter(F.col("eval_set") == "prior") \
    .select("order_id", "user_id")

user_products = prior.join(
    prior_orders,
    "order_id"
)

interactions = user_products.groupBy(
    "user_id",
    "product_id"
).agg(
    F.count("*").alias("purchases")
)

print("Пример взаимодействий:")
interactions.show(20)

print("\nИстория пользователя 1:")

user_history = interactions \
    .filter(F.col("user_id") == 1) \
    .join(products, "product_id") \
    .select("product_name", "purchases") \
    .orderBy(F.desc("purchases"))

user_history.show(20, truncate=False)

spark.stop()
