Chart.defaults.color = "#8b92a0";
Chart.defaults.borderColor = "#252a35";

function makeChart(id, config) {
    const element = document.getElementById(id);

    if (!element) {
        console.error("Canvas not found:", id);
        return;
    }

    new Chart(element, config);
}


makeChart("recommendationChart", {
    type: "bar",
    data: {
        labels: ["Precision@10", "Recall@10", "NDCG@10"],
        datasets: [
            {
                label: "Popularity",
                backgroundColor: "rgba(100, 116, 139, 0.75)",
                borderColor: "#94a3b8",
                borderWidth: 2,
                data: [
                    modelData.popularity.precision,
                    modelData.popularity.recall,
                    modelData.popularity.ndcg
                ]
            },
            {
                label: "ALS",
                backgroundColor: "rgba(124, 92, 255, 0.75)",
                borderColor: "#8b72ff",
                borderWidth: 2,
                data: [
                    modelData.als.precision,
                    modelData.als.recall,
                    modelData.als.ndcg
                ]
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: "bottom"
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});


makeChart("reorderChart", {
    type: "bar",
    data: {
        labels: ["Precision", "Recall", "F1", "Accuracy"],
        datasets: [{
            backgroundColor: "rgba(124, 92, 255, 0.65)",
            borderColor: "#8b72ff",
            borderWidth: 2,
            label: "Random Forest",
            data: [
                modelData.reorder.precision,
                modelData.reorder.recall,
                modelData.reorder.f1,
                modelData.reorder.accuracy
            ]
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});


makeChart("productsChart", {
    type: "bar",
    data: {
        labels: topProducts.map(item => item[0]),
        datasets: [{
            backgroundColor: "rgba(124, 92, 255, 0.65)",
            borderColor: "#8b72ff",
            borderWidth: 2,
            label: "Purchases",
            data: topProducts.map(item => item[1])
        }]
    },
    options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            x: {
                beginAtZero: true
            }
        }
    }
});


makeChart("hourChart", {
    type: "line",
    data: {
        labels: Array.from(
            {length: 24},
            (_, i) => `${i}:00`
        ),
        datasets: [{
            backgroundColor: "rgba(124, 92, 255, 0.65)",
            borderColor: "#8b72ff",
            borderWidth: 2,
            label: "Orders",
            data: ordersByHour,
            tension: 0.35,
            fill: true
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});


makeChart("dayChart", {
    type: "bar",
    data: {
        labels: [
            "Вс",
            "Пн",
            "Вт",
            "Ср",
            "Чт",
            "Пт",
            "Сб"
        ],
        datasets: [{
            backgroundColor: "rgba(124, 92, 255, 0.65)",
            borderColor: "#8b72ff",
            borderWidth: 2,
            label: "Orders",
            data: ordersByDay
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});


function toggleExplanation(element) {
    element.classList.toggle("expanded");
}


async function generateRecommendations() {
    const userId = document.getElementById("userId").value;

    const alsBox = document.getElementById("alsRecommendations");
    const reorderBox = document.getElementById("reorderRecommendations");

    alsBox.innerHTML = `
        <div class="empty">Loading ALS recommendations...</div>
    `;

    reorderBox.innerHTML = `
        <div class="empty">Calculating reorder probabilities...</div>
    `;

    try {
        const [customerResponse, alsResponse, reorderResponse] = await Promise.all([
            fetch(`/api/customer/${userId}`),
            fetch(`/api/recommendations/${userId}`),
            fetch(`/api/reorder/${userId}`)
        ]);

        const customerData = await customerResponse.json();
        const alsData = await alsResponse.json();
        const reorderData = await reorderResponse.json();

        if (!customerResponse.ok) {
            throw new Error(
                customerData.error || "Could not load customer profile"
            );
        }

        document.getElementById("customerProfileTitle").textContent =
            `User #${customerData.user_id}`;

        document.getElementById("customerProfileContent").innerHTML = `
            <div class="customer-stat">
                <span>Previous orders</span>
                <strong>${customerData.orders}</strong>
            </div>

            <div class="customer-stat">
                <span>Unique products</span>
                <strong>${customerData.unique_products}</strong>
            </div>

            <div class="customer-stat">
                <span>Total purchases</span>
                <strong>${customerData.purchases}</strong>
            </div>

            <div class="customer-stat">
                <span>Reorder rate</span>
                <strong>${customerData.reorder_rate.toFixed(1)}%</strong>
            </div>

            <div class="customer-stat customer-stat-wide">
                <span>Favorite department</span>
                <strong>${customerData.favorite_department}</strong>
                <div class="preference-row">
                    <div class="preference-bar">
                        <div style="width: ${customerData.favorite_department_share}%"></div>
                    </div>
                    <small>${customerData.favorite_department_share.toFixed(1)}% покупок</small>
                </div>
            </div>

            <div class="customer-stat customer-stat-wide">
                <span>Favorite aisle</span>
                <strong>${customerData.favorite_aisle}</strong>
                <div class="preference-row">
                    <div class="preference-bar">
                        <div style="width: ${customerData.favorite_aisle_share}%"></div>
                    </div>
                    <small>${customerData.favorite_aisle_share.toFixed(1)}% покупок</small>
                </div>
            </div>

            <div class="customer-stat customer-stat-wide">
                <span>Любимый день для заказов</span>
                <strong>${["Вс", "Пн", "Вт", "Ср", "Чт", "Пт", "Сб"][customerData.favorite_order_day]}</strong>
                <div class="preference-row">
                    <div class="preference-bar">
                        <div style="width: ${customerData.favorite_order_day_share}%"></div>
                    </div>
                    <small>${customerData.favorite_order_day_share.toFixed(1)}% заказов</small>
                </div>
            </div>

            <div class="customer-stat customer-stat-wide">
                <span>Обычное время заказа</span>
                <strong>${String(customerData.favorite_order_hour).padStart(2, "0")}:00–${String((customerData.favorite_order_hour + 1) % 24).padStart(2, "0")}:00</strong>
                <div class="preference-row">
                    <div class="preference-bar">
                        <div style="width: ${customerData.favorite_order_hour_share}%"></div>
                    </div>
                    <small>${customerData.favorite_order_hour_share.toFixed(1)}% заказов</small>
                </div>
            </div>
        `;


        if (!alsResponse.ok) {
            throw new Error(
                alsData.error || "Could not generate ALS recommendations"
            );
        }

        if (!reorderResponse.ok) {
            throw new Error(
                reorderData.error || "Could not generate reorder predictions"
            );
        }

        if (alsData.recommendations.length) {
            const alsInfo = `
                <div class="als-model-info">
                    <strong>Как формируются рекомендации</strong>
                    <p>
                        ALS анализирует скрытые закономерности во взаимодействиях
                        пользователя с товарами. ALS score используется для
                        ранжирования рекомендаций и не является вероятностью покупки.
                    </p>
                </div>
            `;

            alsBox.innerHTML = alsInfo + alsData.recommendations.map(
                (item, index) => `
                    <div class="product-item explainable-product"
                         onclick="toggleExplanation(this)">

                        <div class="product-main-row">
                            <span class="product-rank">${index + 1}</span>

                            <span class="product-name">
                                ${item.product_name}
                                <small class="explain-hint">Click to explain</small>
                            </span>

                            <span class="score score-block">
                                <small class="score-label">ALS score</small>
                                ${item.score.toFixed(3)}
                            </span>

                            <span class="explain-arrow">⌄</span>
                        </div>

                        <div class="product-explanation">
                            <div class="explanation-title">
                                WHY RECOMMENDED?
                            </div>

                            <div class="als-explanation">
                                <div>
                                    <span>ALS score</span>
                                    <strong>${item.score.toFixed(3)}</strong>
                                </div>

                                <div>
                                    <span>Latent factors</span>
                                    <strong>20</strong>
                                </div>
                            </div>

                        </div>

                    </div>
                `
            ).join("");
        } else {
            alsBox.innerHTML = `
                <div class="empty">No ALS recommendations found.</div>
            `;
        }

        if (reorderData.predictions.length) {
            reorderBox.innerHTML = reorderData.predictions.map(
                (item, index) => `
                    <div class="product-item explainable-product"
                         onclick="toggleExplanation(this)">

                        <div class="product-main-row">
                            <span class="product-rank">${index + 1}</span>

                            <span class="product-name">
                                ${item.product_name}
                                <small class="explain-hint">Click to explain</small>
                            </span>

                            <span class="score score-block">
                                <small class="score-label">
                                    Reorder probability
                                </small>
                                ${(item.probability * 100).toFixed(1)}%
                            </span>

                            <span class="explain-arrow">⌄</span>
                        </div>

                        <div class="product-explanation">
                            <div class="explanation-title">
                                WHY REORDER?
                            </div>

                            <div class="explanation-grid">
                                <div>
                                    <span>Previous purchases</span>
                                    <strong>
                                        ${item.explanation.product_purchases}×
                                    </strong>
                                </div>

                                <div>
                                    <span>Purchase rate</span>
                                    <strong>
                                        ${(item.explanation.purchase_rate * 100).toFixed(1)}%
                                    </strong>
                                </div>

                                <div>
                                    <span>Orders since last purchase</span>
                                    <strong>
                                        ${item.explanation.orders_since_last_purchase}
                                    </strong>
                                </div>

                                <div>
                                    <span>Avg. cart position</span>
                                    <strong>
                                        ${item.explanation.avg_cart_position.toFixed(1)}
                                    </strong>
                                </div>
                            </div>

                        </div>

                    </div>
                `
            ).join("");
        } else {
            reorderBox.innerHTML = `
                <div class="empty">
                    No products above the 0.20 threshold.
                </div>
            `;
        }

    } catch (error) {
        alsBox.innerHTML = `
            <div class="empty">${error.message}</div>
        `;

        reorderBox.innerHTML = `
            <div class="empty">${error.message}</div>
        `;

        console.error(error);
    }
}


function updatePipelineTimeline(percent) {

    const stages = [
        { id: "stepDataset", start: 0 },
        { id: "stepAnalytics", start: 25 },
        { id: "stepAls", start: 50 },
        { id: "stepRf", start: 72 },
        { id: "stepResults", start: 94 }
    ];

    stages.forEach((stage, index) => {

        const element = document.getElementById(stage.id);

        if (!element) {
            return;
        }

        element.classList.remove("active", "completed");

        const nextStage = stages[index + 1];

        if (nextStage && percent >= nextStage.start) {
            element.classList.add("completed");
        } else if (percent >= stage.start) {
            element.classList.add("active");
        }

        if (percent >= 100) {
            element.classList.remove("active");
            element.classList.add("completed");
        }
    });
}


const analysisForm = document.getElementById("analysisForm");

if (analysisForm) {
    analysisForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const progress = document.getElementById("pipelineProgress");
        const bar = document.getElementById("progressBar");
        const text = document.getElementById("progressText");
        const percent = document.getElementById("progressPercent");
        const button = analysisForm.querySelector("button");

        progress.style.display = "block";
        button.disabled = true;
        button.textContent = "Pipeline is running...";

        bar.style.width = "0%";
        percent.textContent = "0%";
        text.textContent = "Starting pipeline...";

        const formData = new FormData(analysisForm);

        const pipelineRequest = fetch(
            analysisForm.action,
            {
                method: "POST",
                body: formData
            }
        );

        const progressTimer = setInterval(async () => {
            try {
                const response = await fetch(
                    `/api/progress?t=${Date.now()}`
                );

                const data = await response.json();

                bar.style.width = `${data.percent}%`;
                percent.textContent = `${data.percent}%`;
                text.textContent = data.message;

                updatePipelineTimeline(data.percent);

                if (
                    data.status === "completed" ||
                    data.status === "error"
                ) {
                    clearInterval(progressTimer);
                }
            } catch (error) {
                console.error("Progress update failed:", error);
            }
        }, 1000);

        try {
            await pipelineRequest;

            clearInterval(progressTimer);

            const finalResponse = await fetch(
                `/api/progress?t=${Date.now()}`
            );

            const finalData = await finalResponse.json();

            bar.style.width = `${finalData.percent}%`;
            percent.textContent = `${finalData.percent}%`;
            text.textContent = finalData.message;

            if (finalData.status === "completed") {
                setTimeout(() => {
                    window.location.reload();
                }, 900);
            } else {
                button.disabled = false;
                button.textContent = "Run Pipeline Again";
            }

        } catch (error) {
            clearInterval(progressTimer);

            text.textContent = "Pipeline request failed";
            percent.textContent = "Error";

            button.disabled = false;
            button.textContent = "Run Pipeline Again";

            console.error(error);
        }
    });
}


const lastRunElement = document.getElementById("lastRunTime");

if (lastRunElement) {

    const rawDate = lastRunElement.textContent.trim();
    const date = new Date(rawDate);

    if (!Number.isNaN(date.getTime())) {

        const formattedDate = date.toLocaleString(
            "en-GB",
            {
                day: "2-digit",
                month: "short",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit"
            }
        );

        lastRunElement.textContent = formattedDate;
    }
}
