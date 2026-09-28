from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
import math

spark = SparkSession.builder \
    .appName("EvaluateModelsNDCG") \
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


# Берем пользователей для тестирования
test_users = orders.filter(
    (F.col("eval_set") == "train") &
    (F.col("user_id") <= 5000)
).select(
    "user_id",
    "order_id"
)


# Предыдущие заказы этих пользователей
prior_orders = orders.filter(
    (F.col("eval_set") == "prior") &
    (F.col("user_id") <= 5000)
).select(
    "order_id",
    "user_id"
)


# История покупок
history = prior.join(
    prior_orders,
    "order_id"
)


# Сколько раз пользователь покупал каждый товар
interactions = history.groupBy(
    "user_id",
    "product_id"
).agg(
    F.count("*").alias("purchases")
)


# Реальный следующий заказ
actual = train.join(
    test_users,
    "order_id"
).groupBy(
    "user_id"
).agg(
    F.collect_set("product_id").alias("actual")
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
    nonnegative=True,
    seed=42
)


model = als.fit(interactions)

print("ALS обучена")


# TOP-10 рекомендаций ALS
als_recommendations = model.recommendForAllUsers(10).select(
    "user_id",
    F.expr(
        "transform(recommendations, x -> x.product_id)"
    ).alias("predicted")
)


als_eval = als_recommendations.join(
    actual,
    "user_id"
)


# TOP-10 самых популярных товаров
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
    F.array(
        *[F.lit(x) for x in popular_list]
    )
)


# Precision, Recall и NDCG
def calculate_metrics(rows):

    precision_sum = 0
    recall_sum = 0
    ndcg_sum = 0
    count = 0

    for row in rows:

        predicted = row["predicted"]
        actual_items = set(row["actual"])

        hits = len(
            set(predicted) & actual_items
        )

        precision_sum += hits / 10

        recall_sum += (
            hits / len(actual_items)
        )

        # DCG
        dcg = 0

        for i, product_id in enumerate(predicted):

            if product_id in actual_items:

                dcg += (
                    1 / math.log2(i + 2)
                )

        # идеальный DCG
        ideal_hits = min(
            len(actual_items),
            10
        )

        idcg = sum(
            1 / math.log2(i + 2)
            for i in range(ideal_hits)
        )

        if idcg > 0:
            ndcg_sum += dcg / idcg

        count += 1


    return (
        precision_sum / count,
        recall_sum / count,
        ndcg_sum / count
    )


print("Считаем метрики ALS...")


als_metrics = calculate_metrics(
    als_eval.select(
        "predicted",
        "actual"
    ).toLocalIterator()
)


print("Считаем метрики Popularity...")


pop_metrics = calculate_metrics(
    popularity_eval.select(
        "predicted",
        "actual"
    ).toLocalIterator()
)


print("\nРезультаты:")
print(
    "----------------------------------------------------"
)

print(
    "Model        Precision@10  Recall@10  NDCG@10"
)

print(
    "----------------------------------------------------"
)


print(
    "Popularity   ",
    round(pop_metrics[0], 4),
    "       ",
    round(pop_metrics[1], 4),
    "    ",
    round(pop_metrics[2], 4)
)


print(
    "ALS          ",
    round(als_metrics[0], 4),
    "       ",
    round(als_metrics[1], 4),
    "    ",
    round(als_metrics[2], 4)
)


print(
    "----------------------------------------------------"
)


spark.stop()
