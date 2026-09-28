from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier


DATA = Path("uploads")
MODEL_PATH = Path("models") / "reorder_web"

spark = SparkSession.builder \
    .appName("ReorderWebModel") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Loading data...")

orders = spark.read.csv(
    str(DATA / "orders.csv"),
    header=True,
    inferSchema=True
)

prior = spark.read.csv(
    str(DATA / "order_products__prior.csv"),
    header=True,
    inferSchema=True
)

train = spark.read.csv(
    str(DATA / "order_products__train.csv"),
    header=True,
    inferSchema=True
)

users = orders.filter(
    (F.col("eval_set") == "train") &
    (F.col("user_id") <= 5000)
).select(
    "user_id",
    F.col("order_id").alias("train_order_id")
)

prior_orders = orders.filter(
    (F.col("eval_set") == "prior") &
    (F.col("user_id") <= 5000)
).select(
    "order_id",
    "user_id",
    "order_number"
)

history = prior.join(
    prior_orders,
    "order_id"
)

user_features = prior_orders.groupBy(
    "user_id"
).agg(
    F.max("order_number").alias("user_orders")
)

user_product = history.groupBy(
    "user_id",
    "product_id"
).agg(
    F.count("*").alias("product_purchases"),
    F.min("order_number").alias("first_order"),
    F.max("order_number").alias("last_order"),
    F.avg("add_to_cart_order").alias("avg_cart_position")
)

dataset = user_product.join(
    user_features,
    "user_id"
)

dataset = dataset.withColumn(
    "purchase_rate",
    F.col("product_purchases") / F.col("user_orders")
).withColumn(
    "orders_since_last_purchase",
    F.col("user_orders") - F.col("last_order")
)

actual = train.join(
    users,
    train.order_id == users.train_order_id
).select(
    "user_id",
    "product_id"
).withColumn(
    "label",
    F.lit(1.0)
)

dataset = dataset.join(
    actual,
    ["user_id", "product_id"],
    "left"
).fillna(
    {"label": 0.0}
)

columns = [
    "product_purchases",
    "first_order",
    "last_order",
    "avg_cart_position",
    "user_orders",
    "purchase_rate",
    "orders_since_last_purchase"
]

assembler = VectorAssembler(
    inputCols=columns,
    outputCol="features"
)

training_data = assembler.transform(dataset)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=8,
    seed=42
)

print("Training Random Forest...")

model = rf.fit(training_data)

model.write().overwrite().save(
    str(MODEL_PATH)
)

print("Random Forest trained")
print("Model saved to:", MODEL_PATH)
print("Decision threshold: 0.20")

spark.stop()
