from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession.builder \
    .appName("ReorderPrediction") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

orders = spark.read.csv("data/orders.csv", header=True, inferSchema=True)
prior = spark.read.csv("data/order_products__prior.csv", header=True, inferSchema=True)
train = spark.read.csv("data/order_products__train.csv", header=True, inferSchema=True)

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

history = prior.join(prior_orders, "order_id")

user_features = prior_orders.groupBy("user_id").agg(
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

features = user_product.join(user_features, "user_id")

features = features.withColumn(
    "purchase_rate",
    F.col("product_purchases") / F.col("user_orders")
).withColumn(
    "orders_since_last_purchase",
    F.col("user_orders") - F.col("last_order")
)

actual_products = train.join(
    users,
    train.order_id == users.train_order_id
).select(
    "user_id",
    "product_id"
).withColumn(
    "label",
    F.lit(1.0)
)

dataset = features.join(
    actual_products,
    ["user_id", "product_id"],
    "left"
).fillna({"label": 0.0})

train_users, test_users = users.select("user_id").randomSplit(
    [0.8, 0.2],
    seed=42
)

train_data = dataset.join(train_users, "user_id")
test_data = dataset.join(test_users, "user_id")

feature_columns = [
    "product_purchases",
    "first_order",
    "last_order",
    "avg_cart_position",
    "user_orders",
    "purchase_rate",
    "orders_since_last_purchase"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

print("Обучение модели...")

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=8,
    seed=42
)

model = rf.fit(train_data)

print("Модель обучена")

predictions = model.transform(test_data)

metrics = predictions.agg(
    F.sum(
        F.when(
            (F.col("prediction") == 1) & (F.col("label") == 1), 1
        ).otherwise(0)
    ).alias("TP"),

    F.sum(
        F.when(
            (F.col("prediction") == 1) & (F.col("label") == 0), 1
        ).otherwise(0)
    ).alias("FP"),

    F.sum(
        F.when(
            (F.col("prediction") == 0) & (F.col("label") == 1), 1
        ).otherwise(0)
    ).alias("FN"),

    F.sum(
        F.when(
            (F.col("prediction") == 0) & (F.col("label") == 0), 1
        ).otherwise(0)
    ).alias("TN")
).collect()[0]

tp = metrics["TP"]
fp = metrics["FP"]
fn = metrics["FN"]
tn = metrics["TN"]

precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0

if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0

accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\nРезультаты:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Accuracy:", round(accuracy, 4))

spark.stop()
