"""
Alert Engine: Manages alert queues, deduplication, and export of equipment health matrices.
"""

from dataclasses import asdict
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.agent import MaintenanceTicket
from src.config import (
    ALERTS_LOG_PATH,
    EQUIPMENT_STATUS_PATH,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)


class AlertEngine:
    """Stores, filters, deduplicates, and logs maintenance tickets."""

    def __init__(self):
        self.tickets: List[MaintenanceTicket] = []

    def log_tickets(self, tickets: List[MaintenanceTicket]) -> None:
        """Append newly generated tickets."""
        self.tickets.extend(tickets)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert tickets to pandas DataFrame."""
        if not self.tickets:
            return pd.DataFrame(
                columns=[
                    "ticket_id",
                    "timestamp",
                    "equipment_id",
                    "product_type",
                    "risk_tier",
                    "health_score",
                    "failure_probability_pct",
                    "anomaly_score_pct",
                    "diagnosed_failure_mode",
                    "root_cause_explanation",
                    "recommended_action",
                    "urgency_level",
                ]
            )
        return pd.DataFrame([asdict(t) for t in self.tickets])

    def export_alerts(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """Export alerts log to CSV."""
        path = filepath or ALERTS_LOG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        df_alerts = self.to_dataframe()
        df_alerts.to_csv(path, index=False)
        logger.info(f"Exported {len(df_alerts)} maintenance alerts to {path}")
        return df_alerts

    @staticmethod
    def generate_equipment_status(
        telemetry_df: pd.DataFrame,
        filepath: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Aggregate latest operational health metrics per equipment for Power BI fleet dashboard.
        """
        path = filepath or EQUIPMENT_STATUS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        # Sort by timestamp/udi and grab latest reading per equipment
        latest_df = (
            telemetry_df.sort_values(by="udi")
            .groupby("product_id")
            .last()
            .reset_index()
        )

        cols_to_keep = [
            "product_id",
            "product_type",
            "timestamp",
            "air_temp_k",
            "process_temp_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min",
            "failure_probability_pct",
            "anomaly_score_pct",
            "health_score",
            "risk_tier",
        ]
        available_cols = [c for c in cols_to_keep if c in latest_df.columns]
        status_df = latest_df[available_cols].copy()

        status_df.to_csv(path, index=False)
        logger.info(f"Exported fleet equipment status for {len(status_df)} machines to {path}")
        return status_df
