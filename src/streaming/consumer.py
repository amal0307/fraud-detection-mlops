"""Reads transactions from Kafka and scores them via the API."""
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv
from kafka import KafkaConsumer

load_dotenv(override=True)

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
API_URL = f"http://localhost:{os.getenv('API_PORT', '8000')}/predict"
N_WORKERS = int(os.getenv("CONSUMER_WORKERS", "8"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

http = httpx.Client(timeout=10)


def score(payload: dict):
    """Send one transaction to the API."""
    try:
        r = http.post(API_URL, json=payload)
        r.raise_for_status()
    except Exception as e:
        logger.warning(f"Score failed: {e}")


def main():
    logger.info(f"Connecting to Kafka at {BOOTSTRAP}, topic '{TOPIC}'")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="fraud-scorer",
        enable_auto_commit=True,
    )

    # Concurrent scoring — multiple in-flight requests improve throughput
    executor = ThreadPoolExecutor(max_workers=N_WORKERS)
    received = 0
    start = time.time()
    last_report = start

    try:
        for msg in consumer:
            executor.submit(score, msg.value)
            received += 1

            now = time.time()
            if now - last_report >= 5:
                rate = received / (now - start)
                logger.info(f"Received {received} msgs ({rate:.1f}/sec)")
                last_report = now

    except KeyboardInterrupt:
        logger.info(f"Stopping consumer. Total received: {received}")
    finally:
        executor.shutdown(wait=True)
        consumer.close()


if __name__ == "__main__":
    main()