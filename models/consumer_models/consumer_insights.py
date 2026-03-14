"""
epoch/models/consumer_models/consumer_insights.py
--------------------------------------------------
Consumer-side unified insight model.

Input  : product name + customer pincode
Output : delivery insights — delay risk, estimated days,
         recommended shipping mode, region context

STATUS : STUB — heuristic logic active.
         Replace the three _predict_* functions with your trained
         delay_model.pkl, risk_model.pkl once ready.
         The input/output contract is fixed.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


# ── Model loaders (uncomment when .pkl files are ready) ──────────────────────

_DELAY_MODEL_CACHE = None

def _load_delay_model():
    global _DELAY_MODEL_CACHE
    if _DELAY_MODEL_CACHE is not None:
        return _DELAY_MODEL_CACHE

    candidate_paths = [
        os.path.join(ROOT, "supply_chain_xgb_model.pkl"),
        os.path.join(ROOT, "saved_models", "delay_model.pkl"),
    ]
    for model_path in candidate_paths:
        if os.path.exists(model_path):
            try:
                _DELAY_MODEL_CACHE = joblib.load(model_path)
                return _DELAY_MODEL_CACHE
            except Exception:
                continue
    return None

def _load_risk_model():
    # import joblib
    # return joblib.load(os.path.join(ROOT, "saved_models", "risk_model.pkl"))
    return None


# ── Stub predictors (replace these blocks with your model calls) ──────────────

def _predict_delay_risk(subset: pd.DataFrame) -> dict:
    """
    ── REPLACE WITH YOUR delay_model.pkl ──

    Input : filtered DataFrame rows matching product + pincode
    Output: dict with delay_risk (float 0–1), is_late_predicted (bool)
    """
    if subset.empty:
        return {"delay_risk": 0.5, "is_late_predicted": False, "confidence": 50}

    model = _load_delay_model()

    feature_candidates = [
        "Days_for_shipping_real",
        "Days_for_shipment_scheduled",
        "Order_Item_Quantity",
        "Sales",
        "Order_Profit_Per_Order",
        "Late_delivery_risk",
        "Shipping_Mode",
        "Category_Name",
        "Order_Region",
        "Market",
        "Order_Status",
        "Delivery_Status",
    ]

    def _heuristic_result() -> dict:
        late_rate = float(subset["Late_delivery_risk"].mean()) if "Late_delivery_risk" in subset.columns else 0.5
        is_late = late_rate >= 0.5
        confidence = int(abs(late_rate - 0.5) * 200)
        return {
            "delay_risk": round(late_rate, 3),
            "is_late_predicted": is_late,
            "confidence": confidence,
        }

    if model is None:
        return _heuristic_result()

    available_cols = [c for c in feature_candidates if c in subset.columns]
    if not available_cols:
        return _heuristic_result()

    x_raw = subset[available_cols].copy()

    for col in x_raw.columns:
        if pd.api.types.is_numeric_dtype(x_raw[col]):
            x_raw[col] = pd.to_numeric(x_raw[col], errors="coerce").fillna(0)
        else:
            x_raw[col] = x_raw[col].astype(str).fillna("Unknown")

    try:
        if hasattr(model, "feature_names_in_"):
            x_model = pd.get_dummies(x_raw, drop_first=False)
            needed = list(model.feature_names_in_)
            x_model = x_model.reindex(columns=needed, fill_value=0)
        else:
            x_model = x_raw

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_model)
            if prob.ndim == 2 and prob.shape[1] > 1:
                risk = float(np.clip(prob[:, 1].mean(), 0.0, 1.0))
            else:
                risk = float(np.clip(prob.mean(), 0.0, 1.0))
        else:
            pred = model.predict(x_model)
            risk = float(np.clip(np.mean(pred), 0.0, 1.0))

        is_late = risk >= 0.5
        confidence = int(min(100, max(0, abs(risk - 0.5) * 200)))

        return {
            "delay_risk": round(risk, 3),
            "is_late_predicted": is_late,
            "confidence": confidence,
        }
    except Exception:
        return _heuristic_result()


def _predict_estimated_days(subset: pd.DataFrame) -> dict:
    """
    ── REPLACE WITH YOUR model ──

    Returns estimated shipping days (real vs scheduled).
    """
    if subset.empty:
        return {"estimated_days": None, "scheduled_days": None, "avg_gap": None}

    real_days  = subset["Days_for_shipping_real"].dropna()      if "Days_for_shipping_real"          in subset.columns else pd.Series()
    sched_days = subset["Days_for_shipment_scheduled"].dropna() if "Days_for_shipment_scheduled"      in subset.columns else pd.Series()

    return {
        "estimated_days":  round(float(real_days.mean()),  1) if not real_days.empty  else None,
        "scheduled_days":  round(float(sched_days.mean()), 1) if not sched_days.empty else None,
        "avg_gap":         round(float((real_days.mean() - sched_days.mean())), 1)
                           if (not real_days.empty and not sched_days.empty) else None,
    }


def _predict_shipping_mode(subset: pd.DataFrame, delay_risk: float) -> dict:
    """
    ── REPLACE WITH YOUR shipping_recommendation model ──

    Recommends best shipping mode given product + location context.
    """
    if subset.empty or delay_risk is None:
        return {"recommended_mode": "Standard Class", "reason": "Insufficient data for recommendation"}

    if "Shipping_Mode" not in subset.columns:
        return {"recommended_mode": "Standard Class", "reason": "Shipping mode history is unavailable for this context."}

    mode_df = subset.copy()
    mode_df["Shipping_Mode"] = mode_df["Shipping_Mode"].astype(str).str.strip()
    mode_df = mode_df[mode_df["Shipping_Mode"].str.len() > 0]
    if mode_df.empty:
        return {"recommended_mode": "Standard Class", "reason": "Shipping mode history is unavailable for this context."}

    if "Late_delivery_risk" in mode_df.columns:
        delay_series = pd.to_numeric(mode_df["Late_delivery_risk"], errors="coerce").fillna(0)
    else:
        delay_series = pd.Series([float(delay_risk)] * len(mode_df), index=mode_df.index)

    if "Days_for_shipping_real" in mode_df.columns and "Days_for_shipment_scheduled" in mode_df.columns:
        gap_series = pd.to_numeric(mode_df["Days_for_shipping_real"], errors="coerce").fillna(0) - pd.to_numeric(mode_df["Days_for_shipment_scheduled"], errors="coerce").fillna(0)
    else:
        gap_series = pd.Series([0.0] * len(mode_df), index=mode_df.index)

    mode_df["_delay_metric"] = delay_series
    mode_df["_gap_metric"] = gap_series

    grouped = mode_df.groupby("Shipping_Mode", as_index=False).agg(
        delay_rate=("_delay_metric", "mean"),
        avg_gap=("_gap_metric", "mean"),
        samples=("_delay_metric", "count"),
    )

    if grouped.empty:
        return {"recommended_mode": "Standard Class", "reason": "Not enough shipping-mode history for a rule-based comparison."}

    grouped = grouped.sort_values(["delay_rate", "avg_gap", "samples"], ascending=[True, True, False])
    recommended = str(grouped.iloc[0]["Shipping_Mode"])

    alternatives = grouped[grouped["Shipping_Mode"] != recommended].copy().sort_values(["delay_rate", "avg_gap"], ascending=[False, False])

    rec_delay = float(grouped.iloc[0]["delay_rate"])
    rec_gap = float(grouped.iloc[0]["avg_gap"])

    if alternatives.empty:
        reason = (
            f"{recommended} is recommended based on the best observed delay profile "
            f"(delay {round(rec_delay * 100)}%, avg gap {round(rec_gap, 1)} days)."
        )
        return {"recommended_mode": recommended, "reason": reason}

    top_alt = alternatives.iloc[0]
    alt_mode = str(top_alt["Shipping_Mode"])
    alt_delay = float(top_alt["delay_rate"])
    alt_gap = float(top_alt["avg_gap"])

    delayed_modes = []
    for _, row in alternatives.head(3).iterrows():
        delayed_modes.append(
            f"{row['Shipping_Mode']} ({round(float(row['delay_rate']) * 100)}%, gap {round(float(row['avg_gap']), 1)}d)"
        )

    reason = (
        f"{recommended} is recommended because it has the lowest observed delay "
        f"({round(rec_delay * 100)}%, gap {round(rec_gap, 1)}d). "
        f"Other classes are more delay-prone in this context, led by {alt_mode} "
        f"({round(alt_delay * 100)}%, gap {round(alt_gap, 1)}d). "
        f"Compared classes: {', '.join(delayed_modes)}."
    )

    return {"recommended_mode": recommended, "reason": reason}


def _get_region_context(subset: pd.DataFrame) -> dict:
    """Derive region / market context from matching rows."""
    if subset.empty:
        return {"region": "Unknown", "market": "Unknown", "country": "Unknown"}

    region  = subset["Order_Region"].mode().iloc[0]  if "Order_Region"  in subset.columns and not subset["Order_Region"].dropna().empty  else "Unknown"
    market  = subset["Market"].mode().iloc[0]        if "Market"        in subset.columns and not subset["Market"].dropna().empty        else "Unknown"
    country = subset["Order_Country"].mode().iloc[0] if "Order_Country" in subset.columns and not subset["Order_Country"].dropna().empty else "Unknown"

    return {"region": str(region), "market": str(market), "country": str(country)}


# ── Main entry point ──────────────────────────────────────────────────────────

def predict_for_product_pincode(df: pd.DataFrame, product: str, pincode: str) -> dict:
    """
    Unified consumer insight pipeline.

    Args:
        df      : Full cleaned dataset (from preprocessing)
        product : Product name string (from dropdown)
        pincode : Customer zipcode string (from dropdown)

    Returns:
        dict with keys:
          found         – bool, whether data matched
          product       – echoed product name
          pincode       – echoed pincode
          context       – region / market / country
          delay         – delay risk prediction
          delivery      – estimated days
          shipping      – recommended shipping mode
          order_count   – how many historical orders matched
          avg_sales     – average sales value for this product
    """
    # ── Filter by product ──
    product_col = next((c for c in ["Product_Name", "Category_Name"] if c in df.columns), None)
    if not product_col:
        raise ValueError("No product name column found in dataset.")

    subset = df[df[product_col].astype(str).str.lower() == product.lower()]

    # ── Further filter by pincode if column exists ──
    pincode_col = "Customer_Zipcode" if "Customer_Zipcode" in df.columns else None
    subset_pin  = pd.DataFrame()

    if pincode_col and not subset.empty:
        try:
            subset_pin = subset[
                subset[pincode_col].astype(str).str.split(".").str[0] == str(pincode)
            ]
        except Exception:
            subset_pin = pd.DataFrame()

    # Use pincode-filtered if we have rows, otherwise fall back to product-only
    working_subset = subset_pin if not subset_pin.empty else subset
    found          = not subset.empty

    # ── Run predictions ──
    delay_result    = _predict_delay_risk(working_subset)
    delivery_result = _predict_estimated_days(working_subset)
    shipping_result = _predict_shipping_mode(working_subset, delay_result["delay_risk"])
    context         = _get_region_context(working_subset)

    avg_sales = None
    if "Sales" in working_subset.columns and not working_subset.empty:
        avg_sales = round(float(working_subset["Sales"].mean()), 2)

    return {
        "found":       found,
        "product":     product,
        "pincode":     pincode,
        "context":     context,
        "delay":       delay_result,
        "delivery":    delivery_result,
        "shipping":    shipping_result,
        "order_count": int(len(working_subset)),
        "avg_sales":   avg_sales,
        "data_source": "pincode-matched" if not subset_pin.empty else "product-matched",
    }
