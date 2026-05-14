"""Streams transactions into Kafka — simulates real-world traffic."""
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer

from src.training.features import FEATURE_COLS

load_dotenv(override=True)

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
RATE = float(os.getenv("PRODUCER_RATE_PER_SECOND", "20"))
DRIFT_INJECT = os.getenv("DRIFT_INJECT", "false").lower() == "true"
DATA_PATH = "data/raw/creditcard.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def inject_drift(row: dict) -> dict:
    """
    Simulate a data drift scenario: shift several features' distributions.
    Used to demonstrate drift detection firing in real time.
    """
    drifted = row.copy()
    # Shift Amount upward (e.g., a population spending more)
    drifted["amount"] = drifted["amount"] * np.random.uniform(2.0, 4.0)
    # Shift several V features
    for feature in ["v3", "v10", "v14", "v17"]:
        drifted[feature] = drifted[feature] + np.random.uniform(2.0, 5.0)
    return drifted


def main():
    logger.info(f"Connecting to Kafka at {BOOTSTRAP}")
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=10,  # Batch tiny messages — way better throughput
        acks=1,
    )

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower() for c in df.columns]
    df = df.drop(columns=["time", "class"])

    sleep_per_msg = 1.0 / RATE if RATE > 0 else 0
    logger.info(
        f"Producing to topic '{TOPIC}' at ~{RATE} msg/sec. "
        f"Drift injection: {DRIFT_INJECT}"
    )

    sent = 0
    try:
        while True:
            row = df.sample(1).iloc[0].to_dict()
            row = {k: float(v) for k, v in row.items()}  # JSON-safe types

            if DRIFT_INJECT:
                row = inject_drift(row)

            producer.send(TOPIC, row)
            sent += 1

            if sent % 100 == 0:
                logger.info(f"Sent {sent} messages")

            if sleep_per_msg > 0:
                time.sleep(sleep_per_msg)

    except KeyboardInterrupt:
        logger.info(f"Stopping producer. Total sent: {sent}")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()