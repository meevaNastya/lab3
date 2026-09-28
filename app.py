from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pathlib import Path
from werkzeug.utils import secure_filename
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = "bigdata-project"

UPLOAD_FOLDER = Path("uploads")
RESULTS_FOLDER = Path("results")

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

REQUIRED_FILES = {
    "orders.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
    "order_products__prior.csv",
    "order_products__train.csv"
}

REQUIRED_COLUMNS = {
    "orders.csv": {
        "order_id",
        "user_id",
        "eval_set",
        "order_number",
        "order_dow",
        "order_hour_of_day"
    },
    "products.csv": {
        "product_id",
        "product_name",
        "aisle_id",
        "department_id"
    },
    "aisles.csv": {
        "aisle_id",
        "aisle"
    },
    "departments.csv": {
        "department_id",
        "department"
    },
    "order_products__prior.csv": {
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered"
    },
    "order_products__train.csv": {
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered"
    }
}


def validate_dataset():
    errors = {}

    for filename, required_columns in REQUIRED_COLUMNS.items():
        file_path = UPLOAD_FOLDER / filename

        if not file_path.exists():
            errors[filename] = ["Файл отсутствует"]
            continue

        try:
            with open(file_path, "r", encoding="utf-8-sig") as file:
                header = file.readline().strip()

            columns = {
                column.strip().strip('"')
                for column in header.split(",")
            }

            missing = sorted(required_columns - columns)

            if missing:
                errors[filename] = [
                    f"Отсутствует столбец: {column}"
                    for column in missing
                ]

        except Exception as error:
            errors[filename] = [
                f"Не удалось проверить файл: {error}"
            ]

    return {
        "valid": not errors,
        "errors": errors
    }


stats = {
    "orders": "3.42M",
    "users": "206,209",
    "products": "49,688",
    "interactions": "32.4M",
    "reorder_rate": "58.97%",
    "basket": "10.09"
}

models = {
    "popularity": {
        "precision": 7.13,
        "recall": 6.79,
        "ndcg": 9.38
    },
    "als": {
        "precision": 7.33,
        "recall": 9.04,
        "ndcg": 10.27
    },
    "reorder": {
        "precision": 37.25,
        "recall": 47.58,
        "f1": 41.79,
        "accuracy": 86.40
    }
}

top_products = [
    ["Banana", 472565],
    ["Bag of Organic Bananas", 379450],
    ["Organic Strawberries", 264683],
    ["Organic Baby Spinach", 241921],
    ["Organic Hass Avocado", 213584],
    ["Organic Avocado", 176815],
    ["Large Lemon", 152657],
    ["Strawberries", 142951],
    ["Limes", 140627],
    ["Organic Whole Milk", 137905]
]

orders_by_hour = [
    22758, 12398, 7539, 5474, 5527, 9569,
    30529, 91868, 178201, 257812, 288418, 284728,
    272841, 277999, 283042, 283639, 272553, 228795,
    182912, 140569, 104292, 78109, 61468, 40043
]

orders_by_day = [
    600905, 587478, 467260, 436972,
    426339, 453368, 448761
]


def uploaded_files():
    return sorted(
        file.name
        for file in UPLOAD_FOLDER.glob("*.csv")
        if file.name in REQUIRED_FILES
    )


@app.route("/")
def dashboard():
    files = uploaded_files()

    current_stats = stats
    current_top_products = top_products
    current_orders_by_hour = orders_by_hour
    current_orders_by_day = orders_by_day
    training = None
    last_run = None

    result_file = RESULTS_FOLDER / "dashboard.json"

    if result_file.exists():
        try:
            with open(result_file, "r") as file:
                result = json.load(file)

            result_stats = result["stats"]

            current_stats = {
                "orders": f'{result_stats["orders"]:,}',
                "users": f'{result_stats["users"]:,}',
                "products": f'{result_stats["products"]:,}',
                "interactions": f'{result_stats["interactions"]:,}',
                "reorder_rate": f'{result_stats["reorder_rate"]:.2f}%',
                "basket": f'{result_stats["basket"]:.2f}'
            }

            current_top_products = result["top_products"]
            current_orders_by_hour = result["orders_by_hour"]
            current_orders_by_day = result["orders_by_day"]
            training = result.get("training")
            last_run = result.get("completed_at")

        except (OSError, KeyError, ValueError) as error:
            print("Could not load dashboard results:", error)

    return render_template(
        "index.html",
        stats=current_stats,
        models=models,
        top_products=current_top_products,
        orders_by_hour=current_orders_by_hour,
        orders_by_day=current_orders_by_day,
        uploaded_files=files,
        dataset_ready=validate_dataset()["valid"],
        dataset_validation=validate_dataset(),
        training=training,
        last_run=last_run
    )

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")

    if not files or all(file.filename == "" for file in files):
        flash("Select CSV files first.")
        return redirect(url_for("dashboard"))

    uploaded = 0

    for file in files:
        if not file.filename:
            continue

        filename = secure_filename(file.filename)

        if not filename.lower().endswith(".csv"):
            continue

        file.save(UPLOAD_FOLDER / filename)
        uploaded += 1

    flash(f"Uploaded {uploaded} CSV files.")
    return redirect(url_for("dashboard"))


def save_progress(percent, message, status="running"):
    progress = {
        "percent": percent,
        "message": message,
        "status": status
    }

    with open(RESULTS_FOLDER / "progress.json", "w") as file:
        json.dump(progress, file)


@app.route("/api/progress")
def pipeline_progress():
    progress_file = RESULTS_FOLDER / "progress.json"

    if not progress_file.exists():
        return jsonify({
            "percent": 0,
            "message": "Ready",
            "status": "idle"
        })

    try:
        with open(progress_file, "r") as file:
            return jsonify(json.load(file))
    except (OSError, ValueError):
        return jsonify({
            "percent": 0,
            "message": "Ready",
            "status": "idle"
        })


@app.route("/run-analysis", methods=["POST"])
def run_analysis():
    files = set(uploaded_files())

    missing = REQUIRED_FILES - files

    if missing:
        flash(
            "Missing files: " + ", ".join(sorted(missing))
        )
        return redirect(url_for("dashboard"))

    validation = validate_dataset()

    if not validation["valid"]:
        flash(
            "Dataset schema is incompatible. "
            "Check the required columns before running the pipeline."
        )
        return redirect(url_for("dashboard"))

    spark = None

    try:
        save_progress(5, "Starting Spark...")

        spark = SparkSession.builder \
            .appName("InstacartWebAnalysis") \
            .master("local[*]") \
            .config("spark.driver.memory", "6g") \
            .config("spark.sql.shuffle.partitions", "100") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        save_progress(10, "Loading dataset...")

        orders = spark.read.csv(
            str(UPLOAD_FOLDER / "orders.csv"),
            header=True,
            inferSchema=True
        )

        prior = spark.read.csv(
            str(UPLOAD_FOLDER / "order_products__prior.csv"),
            header=True,
            inferSchema=True
        )

        products = spark.read.csv(
            str(UPLOAD_FOLDER / "products.csv"),
            header=True,
            inferSchema=True
        )

        save_progress(25, "Running PySpark analytics...")

        order_count = orders.count()

        user_count = orders.select(
            "user_id"
        ).distinct().count()

        product_count = products.count()

        interaction_count = prior.count()

        reorder_rate_value = prior.agg(
            F.avg("reordered")
        ).first()[0] * 100

        basket_size = prior.groupBy(
            "order_id"
        ).count().agg(
            F.avg("count")
        ).first()[0]

        top = prior.groupBy(
            "product_id"
        ).count().join(
            products.select(
                "product_id",
                "product_name"
            ),
            "product_id"
        ).orderBy(
            F.desc("count")
        ).limit(10).collect()

        hourly = orders.groupBy(
            "order_hour_of_day"
        ).count().collect()

        daily = orders.groupBy(
            "order_dow"
        ).count().collect()

        hourly_dict = {
            row["order_hour_of_day"]: row["count"]
            for row in hourly
        }

        daily_dict = {
            row["order_dow"]: row["count"]
            for row in daily
        }

        save_progress(45, "Analytics completed")

        interactions = prior.join(
            orders.select("order_id", "user_id"),
            "order_id"
        ).filter(
            F.col("user_id") <= 5000
        ).groupBy(
            "user_id",
            "product_id"
        ).agg(
            F.count("*").alias("purchases")
        )

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

        save_progress(50, "Training ALS recommendation model...")

        als_model = als.fit(interactions)

        save_progress(68, "Saving ALS model...")

        als_model.write().overwrite().save(
            str(Path("models") / "als_web")
        )

        save_progress(72, "ALS trained. Preparing Random Forest...")

        train = spark.read.csv(
            str(UPLOAD_FOLDER / "order_products__train.csv"),
            header=True,
            inferSchema=True
        )

        train_users = orders.filter(
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

        history = prior.join(
            prior_orders,
            "order_id"
        )

        user_features = prior_orders.groupBy(
            "user_id"
        ).agg(
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

        reorder_dataset = user_product.join(
            user_features,
            "user_id"
        )

        reorder_dataset = reorder_dataset.withColumn(
            "purchase_rate",
            F.col("product_purchases") / F.col("user_orders")
        ).withColumn(
            "orders_since_last_purchase",
            F.col("user_orders") - F.col("last_order")
        )

        actual_reorders = train.join(
            train_users,
            train.order_id == train_users.train_order_id
        ).select(
            "user_id",
            "product_id"
        ).withColumn(
            "label",
            F.lit(1.0)
        )

        reorder_dataset = reorder_dataset.join(
            actual_reorders,
            ["user_id", "product_id"],
            "left"
        ).fillna(
            {"label": 0.0}
        )

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

        reorder_training = assembler.transform(
            reorder_dataset
        )

        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=50,
            maxDepth=8,
            seed=42
        )

        save_progress(78, "Training Random Forest...")

        rf_model = rf.fit(reorder_training)

        save_progress(94, "Saving Random Forest model...")

        rf_model.write().overwrite().save(
            str(Path("models") / "reorder_web")
        )

        result = {
            "stats": {
                "orders": order_count,
                "users": user_count,
                "products": product_count,
                "interactions": interaction_count,
                "reorder_rate": round(reorder_rate_value, 2),
                "basket": round(basket_size, 2)
            },
            "top_products": [
                [row["product_name"], row["count"]]
                for row in top
            ],
            "orders_by_hour": [
                hourly_dict.get(hour, 0)
                for hour in range(24)
            ],
            "orders_by_day": [
                daily_dict.get(day, 0)
                for day in range(7)
            ],
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "training": {
                "model": "ALS + Random Forest",
                "status": "trained",
                "users": 5000,
                "rank": 20,
                "regParam": 0.1,
                "alpha": 10,
                "maxIter": 5,
                "rfTrees": 50,
                "rfDepth": 8,
                "rfThreshold": 0.20
            }
        }

        save_progress(98, "Saving dashboard results...")

        with open(
            RESULTS_FOLDER / "dashboard.json",
            "w"
        ) as file:
            json.dump(
                result,
                file,
                indent=2
            )

        save_progress(
            100,
            "Pipeline completed successfully",
            "completed"
        )

        flash(
            "PySpark analysis completed. ALS and Random Forest models trained successfully."
        )

    except Exception as error:
        print(error)

        save_progress(
            0,
            "Pipeline failed",
            "error"
        )

        flash("Pipeline failed. Check the terminal for details.")


    return redirect(url_for("dashboard"))


@app.route("/api/recommendations/<int:user_id>")
def recommendations(user_id):
    if user_id < 1 or user_id > 5000:
        return jsonify({
            "error": "The current ALS model was trained for users 1-5000."
        }), 400

    model_path = Path("models") / "als_web"

    if not model_path.exists():
        return jsonify({
            "error": "ALS model is not trained yet."
        }), 404

    spark = None

    try:
        spark = SparkSession.builder \
            .appName("InstacartRecommendations") \
            .master("local[*]") \
            .config("spark.driver.memory", "6g") \
            .config("spark.sql.shuffle.partitions", "100") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        model = ALSModel.load(str(model_path))

        user_df = spark.createDataFrame(
            [(user_id,)],
            ["user_id"]
        )

        recommendations_df = model.recommendForUserSubset(
            user_df,
            10
        )

        row = recommendations_df.first()

        if row is None:
            return jsonify({
                "user_id": user_id,
                "recommendations": []
            })

        product_ids = [
            item["product_id"]
            for item in row["recommendations"]
        ]

        scores = {
            item["product_id"]: float(item["rating"])
            for item in row["recommendations"]
        }

        products_df = spark.read.csv(
            str(UPLOAD_FOLDER / "products.csv"),
            header=True,
            inferSchema=True
        )

        names = products_df.filter(
            F.col("product_id").isin(product_ids)
        ).select(
            "product_id",
            "product_name"
        ).collect()

        name_map = {
            row["product_id"]: row["product_name"]
            for row in names
        }

        recommendations_list = []

        for product_id in product_ids:
            recommendations_list.append({
                "product_id": product_id,
                "product_name": name_map.get(
                    product_id,
                    f"Product {product_id}"
                ),
                "score": round(scores[product_id], 4)
            })

        return jsonify({
            "user_id": user_id,
            "model": "ALS",
            "recommendations": recommendations_list
        })

    except Exception as error:
        print("Recommendation error:", error)

        return jsonify({
            "error": str(error)
        }), 500




@app.route("/api/customer/<int:user_id>")
def customer_profile(user_id):
    if user_id < 1 or user_id > 5000:
        return jsonify({
            "error": "The current demo supports users 1-5000."
        }), 400

    try:
        spark = SparkSession.builder \
            .appName("CustomerProfile") \
            .master("local[*]") \
            .config("spark.driver.memory", "6g") \
            .config("spark.sql.shuffle.partitions", "100") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        orders = spark.read.csv(
            str(UPLOAD_FOLDER / "orders.csv"),
            header=True,
            inferSchema=True
        )

        prior = spark.read.csv(
            str(UPLOAD_FOLDER / "order_products__prior.csv"),
            header=True,
            inferSchema=True
        )

        products = spark.read.csv(
            str(UPLOAD_FOLDER / "products.csv"),
            header=True,
            inferSchema=True
        )

        aisles = spark.read.csv(
            str(UPLOAD_FOLDER / "aisles.csv"),
            header=True,
            inferSchema=True
        )

        departments = spark.read.csv(
            str(UPLOAD_FOLDER / "departments.csv"),
            header=True,
            inferSchema=True
        )

        user_orders = orders.filter(
            (F.col("user_id") == user_id) &
            (F.col("eval_set") == "prior")
        ).select(
            "order_id",
            "order_number",
            "order_dow",
            "order_hour_of_day"
        )

        order_count = user_orders.count()

        favorite_day_row = user_orders.groupBy(
            "order_dow"
        ).count().orderBy(
            F.col("count").desc(),
            F.col("order_dow")
        ).first()

        favorite_hour_row = user_orders.groupBy(
            "order_hour_of_day"
        ).count().orderBy(
            F.col("count").desc(),
            F.col("order_hour_of_day")
        ).first()

        if order_count == 0:
            return jsonify({
                "error": "User history was not found."
            }), 404

        history = prior.join(
            user_orders,
            "order_id"
        )

        basic_stats = history.agg(
            F.count("*").alias("purchases"),
            F.countDistinct("product_id").alias("unique_products"),
            F.avg("reordered").alias("reorder_rate")
        ).first()

        enriched = history.join(
            products.select(
                "product_id",
                "aisle_id",
                "department_id"
            ),
            "product_id"
        )

        favorite_aisle_row = enriched.groupBy(
            "aisle_id"
        ).count().orderBy(
            F.col("count").desc(),
            F.col("aisle_id")
        ).limit(1).join(
            aisles,
            "aisle_id"
        ).select(
            "aisle",
            "count"
        ).first()

        favorite_department_row = enriched.groupBy(
            "department_id"
        ).count().orderBy(
            F.col("count").desc(),
            F.col("department_id")
        ).limit(1).join(
            departments,
            "department_id"
        ).select(
            "department",
            "count"
        ).first()

        return jsonify({
            "user_id": user_id,
            "orders": int(order_count),
            "unique_products": int(basic_stats["unique_products"]),
            "purchases": int(basic_stats["purchases"]),
            "reorder_rate": round(
                float(basic_stats["reorder_rate"]) * 100,
                1
            ),
            "favorite_department": (
                favorite_department_row["department"]
                if favorite_department_row else "Unknown"
            ),
            "favorite_aisle": (
                favorite_aisle_row["aisle"]
                if favorite_aisle_row else "Unknown"
            ),
            "favorite_department_share": round(
                favorite_department_row["count"] /
                basic_stats["purchases"] * 100,
                1
            ),
            "favorite_aisle_share": round(
                favorite_aisle_row["count"] /
                basic_stats["purchases"] * 100,
                1
            ),
            "favorite_order_day": int(
                favorite_day_row["order_dow"]
            ),
            "favorite_order_day_share": round(
                favorite_day_row["count"] /
                order_count * 100,
                1
            ),
            "favorite_order_hour": int(
                favorite_hour_row["order_hour_of_day"]
            ),
            "favorite_order_hour_share": round(
                favorite_hour_row["count"] /
                order_count * 100,
                1
            )
        })

    except Exception as error:
        print("Customer profile error:", error)

        return jsonify({
            "error": str(error)
        }), 500


@app.route("/api/reorder/<int:user_id>")
def reorder_predictions(user_id):
    if user_id < 1 or user_id > 5000:
        return jsonify({
            "error": "The current Random Forest model supports users 1-5000."
        }), 400

    model_path = Path("models") / "reorder_web"

    if not model_path.exists():
        return jsonify({
            "error": "Random Forest model is not trained yet."
        }), 404

    spark = None

    try:
        spark = SparkSession.builder \
            .appName("ReorderPrediction") \
            .master("local[*]") \
            .config("spark.driver.memory", "6g") \
            .config("spark.sql.shuffle.partitions", "100") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        orders = spark.read.csv(
            str(UPLOAD_FOLDER / "orders.csv"),
            header=True,
            inferSchema=True
        )

        prior = spark.read.csv(
            str(UPLOAD_FOLDER / "order_products__prior.csv"),
            header=True,
            inferSchema=True
        )

        products = spark.read.csv(
            str(UPLOAD_FOLDER / "products.csv"),
            header=True,
            inferSchema=True
        )

        user_orders = orders.filter(
            (F.col("eval_set") == "prior") &
            (F.col("user_id") == user_id)
        ).select(
            "order_id",
            "user_id",
            "order_number"
        )

        order_count = user_orders.agg(
            F.max("order_number").alias("user_orders")
        ).first()["user_orders"]

        if order_count is None:
            return jsonify({
                "user_id": user_id,
                "predictions": []
            })

        history = prior.join(
            user_orders,
            "order_id"
        )

        features = history.groupBy(
            "user_id",
            "product_id"
        ).agg(
            F.count("*").alias("product_purchases"),
            F.min("order_number").alias("first_order"),
            F.max("order_number").alias("last_order"),
            F.avg("add_to_cart_order").alias("avg_cart_position")
        )

        features = features.withColumn(
            "user_orders",
            F.lit(order_count)
        ).withColumn(
            "purchase_rate",
            F.col("product_purchases") / F.col("user_orders")
        ).withColumn(
            "orders_since_last_purchase",
            F.col("user_orders") - F.col("last_order")
        )

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

        features = assembler.transform(features)

        model = RandomForestClassificationModel.load(
            str(model_path)
        )

        predictions = model.transform(features).withColumn(
            "reorder_probability",
            vector_to_array("probability")[1]
        )

        predictions = predictions.filter(
            F.col("reorder_probability") >= 0.20
        ).join(
            products.select("product_id", "product_name"),
            "product_id"
        ).orderBy(
            F.col("reorder_probability").desc()
        ).limit(10)

        rows = predictions.select(
            "product_id",
            "product_name",
            "reorder_probability",
            "product_purchases",
            "purchase_rate",
            "orders_since_last_purchase",
            "avg_cart_position"
        ).collect()

        result = [
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "probability": round(
                    float(row["reorder_probability"]),
                    4
                ),
                "explanation": {
                    "product_purchases": int(
                        row["product_purchases"]
                    ),
                    "purchase_rate": round(
                        float(row["purchase_rate"]),
                        4
                    ),
                    "orders_since_last_purchase": int(
                        row["orders_since_last_purchase"]
                    ),
                    "avg_cart_position": round(
                        float(row["avg_cart_position"]),
                        2
                    )
                }
            }
            for row in rows
        ]

        return jsonify({
            "user_id": user_id,
            "model": "Random Forest",
            "threshold": 0.20,
            "predictions": result
        })

    except Exception as error:
        print("Reorder prediction error:", error)

        return jsonify({
            "error": str(error)
        }), 500



if __name__ == "__main__":
    app.run(debug=True, port=5050)
