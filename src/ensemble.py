"""
ResQFlow ML — Ensemble Predictor
Combines LightGBM + CatBoost + GNN predictions for each road edge.

Weights:
  LightGBM  : 0.35  (reliable baseline, fast, always available)
  CatBoost  : 0.35  (categorical road-type awareness)
  GNN       : 0.30  (neighbourhood-aware, pre-computed cache)

Missing models fall back gracefully: weights are re-normalised at runtime
so the ensemble always sums to 1.0 regardless of which models are loaded.

GNN cache:
  The GNN is expensive to run per-request on a 10 000-edge graph.
  Instead, `refresh_gnn_cache(conditions)` pre-computes travel-time
  predictions for ALL edges in one forward pass and stores them in a dict.
  The router reads `gnn_cache[edge_id]` in O(1).
  The cache is refreshed automatically when conditions change.
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble of LightGBM + CatBoost + GNN travel-time predictors.

    Usage
    -----
    ensemble = EnsemblePredictor(lgbm_model, catboost_model, gnn_model,
                                  gnn_graph_data, gnn_edge_ids)
    ensemble.refresh_gnn_cache(conditions)   # once at launch or on condition change
    weight = ensemble.predict_edge(edge, live_features)
    """

    # Model weights — re-normalised at runtime based on what's loaded
    _WEIGHTS = {"lgbm": 0.35, "catboost": 0.35, "gnn": 0.30}

    def __init__(
        self,
        lgbm_model=None,
        catboost_model=None,
        gnn_model=None,
        gnn_graph_data=None,
        gnn_edge_ids: list = None,
    ):
        self.lgbm    = lgbm_model
        self.catboost = catboost_model
        self.gnn     = gnn_model
        self.graph_data = gnn_graph_data
        self.edge_ids   = gnn_edge_ids or []
        self.gnn_cache: Dict[str, float] = {}      # edge_id → travel_time
        self._last_conditions: dict = {}

        available = []
        if lgbm_model:    available.append("LightGBM")
        if catboost_model: available.append("CatBoost")
        if gnn_model:      available.append("GNN")
        logger.info(f"EnsemblePredictor ready: [{', '.join(available)}]")

    # ── GNN cache ─────────────────────────────────────────────────────────────

    def refresh_gnn_cache(self, conditions: dict = None):
        """Pre-compute GNN travel times for all edges under given conditions."""
        if self.gnn is None or self.graph_data is None:
            return

        conditions = conditions or {}
        if conditions == self._last_conditions and self.gnn_cache:
            logger.info("GNN cache still valid (conditions unchanged).")
            return

        try:
            from gnn_builder import precompute_gnn_weights
            self.gnn_cache = precompute_gnn_weights(
                self.gnn, self.graph_data, self.edge_ids, conditions
            )
            self._last_conditions = dict(conditions)
        except Exception as e:
            logger.error(f"GNN cache refresh failed: {e}")
            self.gnn_cache = {}

    # ── Per-edge prediction ───────────────────────────────────────────────────

    def predict_edge(self, edge, live: dict) -> float:
        """
        Return ensemble-weighted travel time (seconds) for one SUMO edge.

        Falls back gracefully:
          - If only LightGBM available → uses LightGBM 100%
          - If LightGBM + CatBoost → 50/50
          - All three → 35/35/30
        """
        preds: dict = {}
        weights_used: dict = {}

        # ── LightGBM ──────────────────────────────────────────────────────────
        if self.lgbm is not None:
            try:
                import numpy as np
                ms   = edge.getSpeed()
                el   = edge.getLength()
                lc   = edge.getLaneNumber()
                sr   = live["mean_speed"] / max(ms, 0.1)
                dn   = live["vehicle_count"] / max(el, 1.0)
                feats = np.array([[
                    live["vehicle_count"], live["mean_speed"],
                    live["occupancy"],     live["waiting_time"],
                    el, ms, lc,
                    live["hour_of_day"],   live["is_peak_hour"],
                    sr, dn, live["weather_factor"],
                ]])
                p = float(self.lgbm.predict(feats)[0])
                preds["lgbm"] = max(p, 0.1)
                weights_used["lgbm"] = self._WEIGHTS["lgbm"]
            except Exception as e:
                logger.debug(f"LGBM predict failed for {edge.getID()}: {e}")

        # ── CatBoost ──────────────────────────────────────────────────────────
        if self.catboost is not None:
            try:
                from catboost_trainer import predict_catboost
                p = predict_catboost(self.catboost, edge, live)
                if p is not None:
                    preds["catboost"] = p
                    weights_used["catboost"] = self._WEIGHTS["catboost"]
            except Exception as e:
                logger.debug(f"CatBoost predict failed for {edge.getID()}: {e}")

        # ── GNN (from cache) ──────────────────────────────────────────────────
        eid = edge.getID()
        if self.gnn_cache and eid in self.gnn_cache:
            preds["gnn"] = self.gnn_cache[eid]
            weights_used["gnn"] = self._WEIGHTS["gnn"]

        # ── Ensemble ──────────────────────────────────────────────────────────
        if not preds:
            # Absolute fallback: length / speed
            return edge.getLength() / max(edge.getSpeed(), 0.1)

        total_w = sum(weights_used.values())
        result  = sum(
            preds[k] * weights_used[k] / total_w
            for k in preds
        )
        return max(result, 0.1)

    # ── Debug info ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "lgbm":      self.lgbm is not None,
            "catboost":  self.catboost is not None,
            "gnn":       self.gnn is not None,
            "gnn_cache_size": len(self.gnn_cache),
            "weights":   self._WEIGHTS,
        }
