from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder \
    .appName("EvaluateModels") \
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

# Берем пользователей, у которых есть train-заказ
test_users = orders \
    .filter(
        (F.col("eval_set") == "train") &
        (F.col("user_id") <= 5000)
    ) \
    .select("user_id", "order_id")

# Исторические заказы
prior_orders = orders \
    .filter(
        (F.col("eval_set") == "prior") &
        (F.col("user_id") <= 5000)
    ) \
    .select("order_id", "user_id")

history = prior.join(
    prior_orders,
    "order_id"
)

# user-item interactions
interactions = history.groupBy(
    "user_id",
    "product_id"
).agg(
    F.count("*").alias("purchases")
)

print("Обучаем ALS...")

als = ALS(
    userCol="user_id",
    itemCol="product_id",
    ratingCol="purchases",
    rank=10,
    maxIter=5,
    regParam=0.1,
    implicitPrefs=True,
    alpha=10,
    coldStartStrategy="drop",
    nonnegative=True
)

model = als.fit(interactions)

print("ALS обучена")

# Реальные товары следующего заказа
actual = train.join(
    test_users,
    "order_id"
).groupBy(
    "user_id"
).agg(
    F.collect_set("product_id").alias("actual")
)

# TOP-10 ALS
als_recommendations = model.recommendForAllUsers(10) \
    .select(
        "user_id",
        F.expr(
            "transform(recommendations, x -> x.product_id)"
        ).alias("predicted")
    )

als_eval = als_recommendations.join(
    actual,
    "user_id"
)

# Сколько товаров угадано
als_eval = als_eval.withColumn(
    "hits",
    F.size(
        F.array_intersect(
            F.col("predicted"),
            F.col("actual")
        )
    )
)

als_eval = als_eval.withColumn(
    "precision",
    F.col("hits") / F.lit(10)
).withColumn(
    "recall",
    F.col("hits") / F.size("actual")
)

als_metrics = als_eval.agg(
    F.avg("precision").alias("precision"),
    F.avg("recall").alias("recall")
).first()

print("\nALS:")
print("Precision@10:", round(als_metrics["precision"], 4))
print("Recall@10:", round(als_metrics["recall"], 4))

# -------------------------
# POPULARITY
# -------------------------

print("\nСчитаем Popularity...")

popular_products = history.groupBy(
    "product_id"
).count().orderBy(
    F.desc("count")
).limit(10)

popular_list = [
    row["product_id"]
    for row in popular_products.collect()
]

popularity_eval = actual.withColumn(
    "predicted",
    F.array(*[F.lit(x) for x in popular_list])
)

popularity_eval = popularity_eval.withColumn(
    "hits",
    F.size(
        F.array_intersect(
            F.col("predicted"),
            F.col("actual")
        )
    )
)

popularity_eval = popularity_eval.withColumn(
    "precision",
    F.col("hits") / F.lit(10)
).withColumn(
    "recall",
    F.col("hits") / F.size("actual")
)

pop_metrics = popularity_eval.agg(
    F.avg("precision").alias("precision"),
    F.avg("recall").alias("recall")
).first()

print("\nPopularity:")
print("Precision@10:", round(pop_metrics["precision"], 4))
print("Recall@10:", round(pop_metrics["recall"], 4))

print("\nСравнение моделей:")
print("--------------------------------")
print("Model        Precision@10  Recall@10")
print("--------------------------------")
print(
    "Popularity   ",
    round(pop_metrics["precision"], 4),
    "       ",
    round(pop_metrics["recall"], 4)
)
print(
    "ALS          ",
    round(als_metrics["precision"], 4),
    "       ",
    round(als_metrics["recall"], 4)
)
print("--------------------------------")

spark.stop()
