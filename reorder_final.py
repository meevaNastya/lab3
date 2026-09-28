from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.functions import vector_to_array

spark = SparkSession.builder \
    .appName("ReorderFinal") \
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

dataset = user_product.join(user_features, "user_id")

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
).fillna({"label": 0.0})

train_users, validation_users, test_users = users.select(
    "user_id"
).randomSplit(
    [0.7, 0.15, 0.15],
    seed=42
)

train_data = dataset.join(train_users, "user_id")
validation_data = dataset.join(validation_users, "user_id")
test_data = dataset.join(test_users, "user_id")

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

train_data = assembler.transform(train_data)
validation_data = assembler.transform(validation_data)
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


def probabilities(data):
    return model.transform(data).select(
        "label",
        vector_to_array("probability")[1].alias("probability")
    ).cache()


def metrics(data, threshold):
    values = data.agg(
        F.sum(
            F.when(
                (F.col("probability") >= threshold) &
                (F.col("label") == 1),
                1
            ).otherwise(0)
        ).alias("tp"),

        F.sum(
            F.when(
                (F.col("probability") >= threshold) &
                (F.col("label") == 0),
                1
            ).otherwise(0)
        ).alias("fp"),

        F.sum(
            F.when(
                (F.col("probability") < threshold) &
                (F.col("label") == 1),
                1
            ).otherwise(0)
        ).alias("fn"),

        F.sum(
            F.when(
                (F.col("probability") < threshold) &
                (F.col("label") == 0),
                1
            ).otherwise(0)
        ).alias("tn")
    ).collect()[0]

    tp = values["tp"]
    fp = values["fp"]
    fn = values["fn"]
    tn = values["tn"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return precision, recall, f1, accuracy


validation_predictions = probabilities(validation_data)

thresholds = [
    0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65,
    0.70
]

best_threshold = 0
best_f1 = 0

print("\nValidation:")
print("Threshold   Precision   Recall      F1")
print("-----------------------------------------")

for threshold in thresholds:
    precision, recall, f1, accuracy = metrics(
        validation_predictions,
        threshold
    )

    print(
        f"{threshold:<11.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nЛучший threshold:", best_threshold)
print("Validation F1:", round(best_f1, 4))

validation_predictions.unpersist()

print("\nПроверяем на test...")

test_predictions = probabilities(test_data)

precision, recall, f1, accuracy = metrics(
    test_predictions,
    best_threshold
)

print("\nФинальный результат:")
print("Threshold:", best_threshold)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Accuracy:", round(accuracy, 4))

test_predictions.unpersist()

spark.stop()
