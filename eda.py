from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("InstacartEDA") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

orders = spark.read.csv("data/orders.csv", header=True, inferSchema=True)
prior = spark.read.csv("data/order_products__prior.csv", header=True, inferSchema=True)
products = spark.read.csv("data/products.csv", header=True, inferSchema=True)

users_count = orders.select("user_id").distinct().count()
print("Уникальных пользователей:", users_count)

basket_sizes = prior.groupBy("order_id").count()
average_basket = basket_sizes.agg(F.avg("count")).first()[0]
print("Среднее количество товаров в заказе:", round(average_basket, 2))

popular = prior.groupBy("product_id") \
    .count() \
    .join(products, "product_id") \
    .select("product_name", "count") \
    .orderBy(F.desc("count"))

print("\n10 самых популярных товаров:")
popular.show(10, truncate=False)

reorder_rate = prior.agg(F.avg("reordered")).first()[0]
print("Доля повторно заказанных товаров:", round(reorder_rate * 100, 2), "%")

print("\nКоличество заказов по часам:")
orders.groupBy("order_hour_of_day") \
    .count() \
    .orderBy("order_hour_of_day") \
    .show(24)

print("\nКоличество заказов по дням недели:")
orders.groupBy("order_dow") \
    .count() \
    .orderBy("order_dow") \
    .show(7)

spark.stop()
