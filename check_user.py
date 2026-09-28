from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("CheckUser") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

orders = spark.read.csv(
    "data/orders.csv",
    header=True,
    inferSchema=True
)

train = spark.read.csv(
    "data/order_products__train.csv",
    header=True,
    inferSchema=True
)

products = spark.read.csv(
    "data/products.csv",
    header=True,
    inferSchema=True
)

user_order = orders \
    .filter(
        (F.col("user_id") == 1) &
        (F.col("eval_set") == "train")
    ) \
    .select("order_id")

actual = train.join(
    user_order,
    "order_id"
).join(
    products,
    "product_id"
).select(
    "product_name",
    "reordered"
)

print("Реальный следующий заказ пользователя 1:")
actual.show(30, truncate=False)

spark.stop()
