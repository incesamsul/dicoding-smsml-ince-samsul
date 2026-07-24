from prometheus_client import Counter, Histogram, start_http_server
import random, time

PREDICTION_COUNT = Counter("model_prediction_total", "Total prediction requests")
PREDICTION_LATENCY = Histogram("model_prediction_latency_seconds", "Prediction latency")

if __name__ == "__main__":
    start_http_server(8000)
    while True:
        with PREDICTION_LATENCY.time():
            time.sleep(random.uniform(0.01, 0.05))
        PREDICTION_COUNT.inc()
        time.sleep(1)
