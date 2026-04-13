"""Pydantic schemas for the Fraud Detection API."""

from pydantic import BaseModel, Field
from typing import Optional


class ScoreRequest(BaseModel):
    tx_id: int = Field(..., description="Transaction node ID to score")

    model_config = {"json_schema_extra": {"example": {"tx_id": 12345}}}


class BatchScoreRequest(BaseModel):
    tx_ids: list[int] = Field(..., description="List of transaction IDs to score")
    model_config = {"json_schema_extra": {"example": {"tx_ids": [12345, 67890, 11111]}}}


class FraudScore(BaseModel):
    tx_id:          int
    gae_score:      float  = Field(..., description="GAE reconstruction error (0–1)")
    sage_prob:      float  = Field(..., description="GraphSAGE fraud probability (0–1)")
    ensemble_score: float  = Field(..., description="Weighted ensemble: 0.4*GAE + 0.6*SAGE")
    risk_label:     str    = Field(..., description="low / medium / high / critical")
    is_flagged:     bool   = Field(..., description="True if above fraud threshold")


class BatchScoreResponse(BaseModel):
    results:        list[FraudScore]
    n_flagged:      int
    flag_rate:      float


class ExplainResponse(BaseModel):
    tx_id:              int
    risk_label:         str
    neighbourhood:      list[dict]     = Field(..., description="k-hop neighbours from Neo4j")
    n_illicit_neighbours: int
    n_licit_neighbours:   int
    risk_reasoning:     str


class DriftStatusResponse(BaseModel):
    drifted:              bool
    affected_time_steps:  list[int]
    ks_statistic:         float
    ks_pvalue:            float
    mean_shift_zscore:    float
    message:              str
    recommendation:       str


class HealthResponse(BaseModel):
    status:   str
    gae_loaded:   bool
    sage_loaded:  bool
    neo4j_connected: bool
    n_nodes_indexed: Optional[int] = None
