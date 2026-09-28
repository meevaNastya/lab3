from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("InstacartAnalysis") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

orders = spark.read.csv("data/orders.csv", header=True, inferSchema=True)
prior = spark.read.csv("data/order_products__prior.csv", header=True, inferSchema=True)
train = spark.read.csv("data/order_products__train.csv", header=True, inferSchema=True)
products = spark.read.csv("data/products.csv", header=True, inferSchema=True)
aisles = spark.read.csv("data/aisles.csv", header=True, inferSchema=True)
departments = spark.read.csv("data/departments.csv", header=True, inferSchema=True)

print("Количество заказов:", orders.count())
print("Товаров в предыдущих заказах:", prior.count())
print("Товаров в обучающей выборке:", train.count())
print("Количество товаров:", products.count())
print("Количество категорий:", aisles.count())
print("Количество отделов:", departments.count())

print("\nТаблица products:")
products.show(5, truncate=False)

print("\nТаблица order_products__prior:")
prior.show(5)

spark.stop()
