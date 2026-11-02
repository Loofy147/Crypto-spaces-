"""PostgreSQL persistence for immutable market observations and forecast records."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.ohlcv_contract import MarketCandle


class PostgresRepository:
    """Small transaction-scoped repository; connection ownership stays explicit."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @classmethod
    def from_dsn(cls, dsn: str | None = None) -> "PostgresRepository":
        connection_string = dsn or os.environ.get("DATABASE_URL")
        if not connection_string:
            raise RuntimeError("DATABASE_URL is not configured")
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("Install psycopg[binary] to use PostgreSQL persistence") from exc
        return cls(psycopg.connect(connection_string, row_factory=dict_row))

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "PostgresRepository":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self._connection.commit()
        else:
            self._connection.rollback()
        self.close()

    @staticmethod
    def _jsonb(value: Any) -> Any:
        try:
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise RuntimeError("Install psycopg[binary] to use PostgreSQL persistence") from exc
        return Jsonb(value)

    def initialize_schema(self) -> None:
        root = Path(__file__).resolve().parents[2]
        migration_path = root / "migrations" / "001_market_data.sql"
        sql = migration_path.read_text(encoding="utf-8")
        self._connection.execute(sql)
        self._connection.commit()

    def ping(self) -> bool:
        row = self._connection.execute("SELECT 1 AS ok").fetchone()
        return bool(row and row["ok"] == 1)

    def upsert_candles(self, candles: Sequence[MarketCandle]) -> int:
        if not candles:
            return 0
        observation_statement = """
            INSERT INTO market_candle_observations (
                venue, instrument, base_currency, quote_currency, interval,
                source_bar_timestamp, event_time, observed_at, open, high, low,
                close, volume, trade_count, is_final, source_payload_sha256
            ) VALUES (
                %(venue)s, %(instrument)s, %(base_currency)s, %(quote_currency)s,
                %(interval)s, %(source_bar_timestamp)s, %(event_time)s,
                %(ingested_at)s, %(open)s, %(high)s, %(low)s, %(close)s,
                %(volume)s, %(trade_count)s, %(is_final)s, %(source_payload_sha256)s
            )
            ON CONFLICT (venue, instrument, interval, event_time, source_payload_sha256)
            DO NOTHING
        """
        statement = """
            INSERT INTO market_candles (
                venue, instrument, base_currency, quote_currency, interval,
                source_bar_timestamp, event_time, open, high, low, close,
                volume, trade_count, ingested_at, is_final,
                source_payload_sha256, source_timestamp_semantics,
                canonical_timestamp_semantics
            ) VALUES (
                %(venue)s, %(instrument)s, %(base_currency)s, %(quote_currency)s,
                %(interval)s, %(source_bar_timestamp)s, %(event_time)s,
                %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s,
                %(trade_count)s, %(ingested_at)s, %(is_final)s,
                %(source_payload_sha256)s, %(source_timestamp_semantics)s,
                %(canonical_timestamp_semantics)s
            )
            ON CONFLICT (venue, instrument, interval, event_time) DO UPDATE SET
                source_bar_timestamp = EXCLUDED.source_bar_timestamp,
                open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                close = EXCLUDED.close, volume = EXCLUDED.volume,
                trade_count = EXCLUDED.trade_count,
                ingested_at = EXCLUDED.ingested_at,
                is_final = EXCLUDED.is_final,
                source_payload_sha256 = EXCLUDED.source_payload_sha256
            WHERE market_candles.ingested_at <= EXCLUDED.ingested_at
        """
        rows = [candle.to_dict() for candle in candles]
        with self._connection.cursor() as cursor:
            cursor.executemany(observation_statement, rows)
            cursor.executemany(statement, rows)
            changed = cursor.rowcount
        self._connection.commit()
        return int(changed)

    def start_market_data_run(self, manifest: Mapping[str, Any]) -> int:
        row = self._connection.execute(
            """
            INSERT INTO market_data_runs (
                provider, requested_instrument, returned_instrument,
                base_currency, quote_currency, interval, requested_start,
                requested_end, retrieved_at, status, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'running', %s)
            RETURNING id
            """,
            (
                manifest["provider"], manifest["requested_instrument"],
                manifest["returned_instrument"], manifest["base_currency"],
                manifest["quote_currency"], manifest["interval"],
                manifest.get("requested_start"), manifest.get("requested_end"),
                manifest["retrieved_at"], self._jsonb({
                    "endpoint": manifest["endpoint"],
                    "source_documentation": manifest["source_documentation"],
                }),
            ),
        ).fetchone()
        self._connection.commit()
        return int(row["id"])

    def finish_market_data_run(
        self,
        run_id: int,
        *,
        status: str,
        row_count: int = 0,
        finalized_count: int = 0,
        raw_response_sha256: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if status not in {"succeeded", "failed"}:
            raise ValueError("market data run status must be succeeded or failed")
        self._connection.execute(
            """
            UPDATE market_data_runs
            SET finished_at = now(), status = %s, row_count = %s,
                finalized_count = %s, raw_response_sha256 = %s,
                error_message = %s
            WHERE id = %s
            """,
            (status, row_count, finalized_count, raw_response_sha256, error_message, run_id),
        )
        self._connection.commit()

    def start_forecast_cycle(self) -> int:
        row = self._connection.execute(
            "INSERT INTO forecast_cycles (status) VALUES ('running') RETURNING id"
        ).fetchone()
        self._connection.commit()
        return int(row["id"])

    def finish_forecast_cycle(
        self, cycle_id: int, *, status: str, forecasts_created: int,
        errors: Sequence[str],
    ) -> None:
        if status not in {"succeeded", "partial", "failed"}:
            raise ValueError("invalid forecast cycle status")
        self._connection.execute(
            """
            UPDATE forecast_cycles
            SET finished_at = now(), status = %s, forecasts_created = %s,
                error_count = %s, errors = %s
            WHERE id = %s
            """,
            (status, forecasts_created, len(errors), self._jsonb(list(errors)), cycle_id),
        )
        self._connection.commit()

    def save_forecast(self, forecast: Mapping[str, Any]) -> None:
        self._connection.execute(
            """
            INSERT INTO volatility_forecasts (
                venue, instrument, base_currency, quote_currency, interval,
                forecast_horizon_steps, forecast_origin, model,
                innovation_distribution, observations, input_sha256,
                model_parameters, log_likelihood, aic, bic,
                conditional_variance, conditional_volatility,
                annualized_volatility, reference_price, mean_log_return,
                degrees_of_freedom, price_quantiles
            ) VALUES (
                %(venue)s, %(instrument)s, %(base_currency)s, %(quote_currency)s,
                %(interval)s, %(forecast_horizon_steps)s, %(forecast_origin)s,
                %(model)s, %(innovation_distribution)s, %(observations)s,
                %(input_sha256)s, %(model_parameters)s, %(log_likelihood)s,
                %(aic)s, %(bic)s, %(conditional_variance)s,
                %(conditional_volatility)s, %(annualized_volatility)s,
                %(reference_price)s, %(mean_log_return)s, %(degrees_of_freedom)s,
                %(price_quantiles)s
            )
            ON CONFLICT (
                venue, instrument, interval, forecast_horizon_steps,
                forecast_origin, model, innovation_distribution
            ) DO UPDATE SET
                generated_at = now(),
                observations = EXCLUDED.observations,
                input_sha256 = EXCLUDED.input_sha256,
                model_parameters = EXCLUDED.model_parameters,
                log_likelihood = EXCLUDED.log_likelihood,
                aic = EXCLUDED.aic, bic = EXCLUDED.bic,
                conditional_variance = EXCLUDED.conditional_variance,
                conditional_volatility = EXCLUDED.conditional_volatility,
                annualized_volatility = EXCLUDED.annualized_volatility,
                reference_price = EXCLUDED.reference_price,
                mean_log_return = EXCLUDED.mean_log_return,
                degrees_of_freedom = EXCLUDED.degrees_of_freedom,
                price_quantiles = EXCLUDED.price_quantiles
            """,
            {
                **forecast,
                "model_parameters": self._jsonb(forecast["model_parameters"]),
                "price_quantiles": self._jsonb(forecast["price_quantiles"]),
            },
        )
        self._connection.commit()

    def latest_forecasts(
        self, base_currency: str = "ETH", quote_currency: str = "USD",
        interval: str = "1h", venue: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        bounded_limit = min(max(int(limit), 1), 200)
        rows = self._connection.execute(
            """
            SELECT * FROM volatility_forecasts
            WHERE base_currency = %s AND quote_currency = %s AND interval = %s
              AND (%s IS NULL OR venue = %s)
            ORDER BY forecast_origin DESC, generated_at DESC, model ASC
            LIMIT %s
            """,
            (base_currency, quote_currency, interval, venue, venue, bounded_limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def latest_market_data_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        bounded_limit = min(max(int(limit), 1), 100)
        rows = self._connection.execute(
            """
            SELECT DISTINCT ON (provider, requested_instrument, interval)
                id, provider, requested_instrument, returned_instrument,
                base_currency, quote_currency, interval, retrieved_at,
                started_at, finished_at, status, row_count, finalized_count,
                raw_response_sha256, error_message
            FROM market_data_runs
            ORDER BY provider, requested_instrument, interval, started_at DESC
            LIMIT %s
            """,
            (bounded_limit,),
        ).fetchall()
        return [dict(row) for row in rows]
