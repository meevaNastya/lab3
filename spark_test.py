from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Instacart") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

orders = spark.read.csv("data/orders.csv", header=True, inferSchema=True)

print("Версия Spark:", spark.version)

print("Структура данных:")
orders.printSchema()

print("Первые 5 строк:")
orders.show(5)

count = orders.count()
print("Количество заказов:", count)

spark.stop()
