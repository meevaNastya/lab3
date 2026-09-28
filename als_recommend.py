from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder \
    .appName("InstacartALS") \
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

data = interactions.filter(F.col("user_id") <= 30000)

print("Данные подготовлены")
print("Начинаем обучение ALS...")

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

model = als.fit(data)

print("Модель обучена")

user = spark.createDataFrame([(1,)], ["user_id"])

recommendations = model.recommendForUserSubset(user, 30)

recommendations = recommendations.select(
    "user_id",
    F.explode("recommendations").alias("rec")
).select(
    "user_id",
    F.col("rec.product_id").alias("product_id"),
    F.col("rec.rating").alias("score")
)

purchased = data \
    .filter(F.col("user_id") == 1) \
    .select("product_id")

new_recommendations = recommendations.join(
    purchased,
    "product_id",
    "left_anti"
)

result = new_recommendations.join(
    products,
    "product_id"
).select(
    "product_name",
    "score"
).orderBy(F.desc("score"))

print("\nНовые рекомендации ALS для пользователя 1:")
result.show(10, truncate=False)

print("\nЧто пользователь 1 покупал раньше:")

history = data \
    .filter(F.col("user_id") == 1) \
    .join(products, "product_id") \
    .select(
        "product_name",
        "purchases"
    ) \
    .orderBy(F.desc("purchases"))

history.show(10, truncate=False)

spark.stop()
