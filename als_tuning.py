from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder \
    .appName("ALSTuning") \
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

users = orders.filter(
    (F.col("eval_set") == "train") &
    (F.col("user_id") <= 5000)
).select("user_id").distinct()

prior_orders = orders.filter(
    F.col("eval_set") == "prior"
).select(
    "order_id",
    "user_id",
    "order_number"
).join(
    users,
    "user_id"
)

last_prior = prior_orders.groupBy("user_id").agg(
    F.max("order_number").alias("validation_order_number")
)

orders_split = prior_orders.join(
    last_prior,
    "user_id"
)

history_orders = orders_split.filter(
    F.col("order_number") < F.col("validation_order_number")
).select(
    "order_id",
    "user_id"
)

validation_orders = orders_split.filter(
    F.col("order_number") == F.col("validation_order_number")
).select(
    "order_id",
    "user_id"
)

history = prior.join(
    history_orders,
    "order_id"
)

interactions = history.groupBy(
    "user_id",
    "product_id"
).agg(
    F.count("*").alias("purchases")
).cache()

validation = prior.join(
    validation_orders,
    "order_id"
).groupBy(
    "user_id"
).agg(
    F.collect_set("product_id").alias("actual")
).cache()

print("Interactions:", interactions.count())
print("Validation users:", validation.count())

params = [
    (5, 0.05),
    (5, 0.10),
    (10, 0.05),
    (10, 0.10),
    (20, 0.05),
    (20, 0.10)
]


def evaluate(recommendations):
    result = recommendations.join(
        validation,
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
        1.0 / __import__("math").log2(i + 2)
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

    return (
        metrics["precision"],
        metrics["recall"],
        metrics["ndcg"]
    )


results = []

print()
print("rank  regParam   Precision@10   Recall@10   NDCG@10")
print("------------------------------------------------------")

for rank, reg_param in params:

    als = ALS(
        userCol="user_id",
        itemCol="product_id",
        ratingCol="purchases",
        rank=rank,
        maxIter=5,
        regParam=reg_param,
        implicitPrefs=True,
        alpha=10,
        coldStartStrategy="drop",
        nonnegative=True,
        seed=42
    )

    model = als.fit(interactions)

    recommendations = model.recommendForAllUsers(10)

    precision, recall, ndcg = evaluate(
        recommendations
    )

    results.append({
        "rank": rank,
        "regParam": reg_param,
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg
    })

    print(
        f"{rank:<6}"
        f"{reg_param:<11.2f}"
        f"{precision:<15.4f}"
        f"{recall:<12.4f}"
        f"{ndcg:.4f}"
    )


best = max(
    results,
    key=lambda x: x["ndcg"]
)

print()
print("Лучшие параметры по NDCG@10:")
print("rank:", best["rank"])
print("regParam:", best["regParam"])
print("alpha: 10")
print("maxIter: 5")

print()
print("Validation metrics:")
print(
    "Precision@10:",
    round(best["precision"], 4)
)
print(
    "Recall@10:",
    round(best["recall"], 4)
)
print(
    "NDCG@10:",
    round(best["ndcg"], 4)
)

interactions.unpersist()
validation.unpersist()

spark.stop()
