from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
import math

spark = SparkSession.builder \
    .appName("ALSFinal") \
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

train = spark.read.csv(
    "data/order_products__train.csv",
    header=True,
    inferSchema=True
)

users = orders.filter(
    (F.col("eval_set") == "train") &
    (F.col("user_id") <= 5000)
).select(
    "user_id",
    F.col("order_id").alias("train_order_id")
).distinct()

prior_orders = orders.filter(
    F.col("eval_set") == "prior"
).select(
    "order_id",
    "user_id"
).join(
    users.select("user_id"),
    "user_id"
)

history = prior.join(
    prior_orders,
    "order_id"
)

interactions = history.groupBy(
    "user_id",
    "product_id"
).agg(
    F.count("*").alias("purchases")
).cache()

actual = train.join(
    users,
    train.order_id == users.train_order_id
).groupBy(
    "user_id"
).agg(
    F.collect_set("product_id").alias("actual")
).cache()

print("Обучение финальной ALS...")

als = ALS(
    userCol="user_id",
    itemCol="product_id",
    ratingCol="purchases",
    rank=20,
    maxIter=5,
    regParam=0.1,
    implicitPrefs=True,
    alpha=10,
    coldStartStrategy="drop",
    nonnegative=True,
    seed=42
)

model = als.fit(interactions)

recommendations = model.recommendForAllUsers(10)

result = recommendations.join(
    actual,
    "user_id"
)

result = result.withColumn(
    "predicted",
    F.expr(
        "transform(recommendations, x -> x.product_id)"
    )
)

result = result.withColumn(
    "hits",
    F.size(
        F.array_intersect(
            F.col("predicted"),
            F.col("actual")
        )
    )
)

result = result.withColumn(
    "precision",
    F.col("hits") / F.lit(10)
)

result = result.withColumn(
    "recall",
    F.when(
        F.size("actual") > 0,
        F.col("hits") / F.size("actual")
    ).otherwise(0)
)

discounts = [
    1.0 / math.log2(i + 2)
    for i in range(10)
]


@F.udf("double")
def ndcg(predicted, actual):
    if not predicted or not actual:
        return 0.0

    actual_set = set(actual)

    dcg = 0.0

    for i, product in enumerate(predicted[:10]):
        if product in actual_set:
            dcg += discounts[i]

    ideal_count = min(len(actual_set), 10)

    if ideal_count == 0:
        return 0.0

    idcg = sum(discounts[:ideal_count])

    return dcg / idcg


result = result.withColumn(
    "ndcg",
    ndcg("predicted", "actual")
)

metrics = result.agg(
    F.avg("precision").alias("precision"),
    F.avg("recall").alias("recall"),
    F.avg("ndcg").alias("ndcg")
).collect()[0]

print()
print("Финальные параметры:")
print("rank: 20")
print("regParam: 0.1")
print("alpha: 10")
print("maxIter: 5")

print()
print("FINAL TEST METRICS")
print("------------------")
print(
    "Precision@10:",
    round(metrics["precision"], 4)
)
print(
    "Recall@10:",
    round(metrics["recall"], 4)
)
print(
    "NDCG@10:",
    round(metrics["ndcg"], 4)
)

interactions.unpersist()
actual.unpersist()

spark.stop()
