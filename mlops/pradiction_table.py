import duckdb
from typing import List
from typing import Dict
from mlops.schema import PredictionTableResponse

class PredictionTable:
    def __init__(self, db_path: str = "prediction.duckdb"):
        self.conn = duckdb.connect(db_path)
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                customer_id TEXT,

                monthly_charges DOUBLE,
                tenure INTEGER,
                total_charges DOUBLE,
                high_value_fiber BOOLEAN,
                churn BOOLEAN,
                
                churn_probability DOUBLE,
                churn_prediction BOOLEAN,
                confidence TEXT,
                model_version TEXT,
                prediction_timestamp TIMESTAMP
            )
        """)

    def insert_prediction(self, prediction: PredictionTableResponse):
        self.conn.execute("""
            INSERT INTO predictions (
                customer_id,
                monthly_charges,
                tenure,
                total_charges,
                high_value_fiber,
                churn,
                churn_probability,
                churn_prediction,
                confidence,
                model_version,
                prediction_timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction.customer_id,
            prediction.monthly_charges,
            prediction.tenure,
            prediction.total_charges,
            prediction.high_value_fiber,
            prediction.churn,
            prediction.churn_probability,
            prediction.churn_prediction,
            prediction.confidence,
            prediction.model_version,
            prediction.prediction_timestamp,
        ))

    def get_latest_predictions(self, limit: int = 100) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT * FROM (
            SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY prediction_timestamp DESC) AS row_num
                FROM predictions
            )
            WHERE row_num = 1
            ORDER BY prediction_timestamp DESC
            LIMIT ?
        """, [limit]).fetchall()

        return [
            {
                "customer_id": row[0],
                
                "monthly_charges": row[1],
                "tenure": row[2],
                "total_charges": row[3],
                "high_value_fiber": row[4],
                "churn": row[5],

                "churn_probability": row[6],
                "churn_prediction": row[7],
                "confidence": row[8],
                "model_version": row[9],
                "prediction_timestamp": row[10],
            }
            for row in rows
        ]
