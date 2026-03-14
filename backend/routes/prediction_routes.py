"""
epoch/backend/routes/prediction_routes.py
------------------------------------------
Endpoints:
  GET  /api/predict/options/products   → unique product names from dataset
  GET  /api/predict/options/cities     → unique destination cities
  GET  /api/predict/options/pincodes   → unique customer zipcodes
  POST /api/predict/demand             → Producer: product + city → demand insights
  POST /api/predict/consumer           → Consumer: product + pincode → delivery insights
"""

import os
import sys
from flask import Blueprint, request, jsonify

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

prediction_bp = Blueprint("prediction", __name__)


def _augment_shipping_reason_with_external_signals(result: dict) -> dict:
    """
    Attach weather/news disruption notes as separate fields when delay risk is elevated.
    Keeps the existing model/rule-based shipping reason unchanged.
    """
    if not isinstance(result, dict):
        return result

    delay = result.get("delay") or {}
    shipping = result.get("shipping") or {}
    context = result.get("context") or {}

    delay_risk = delay.get("delay_risk")
    is_late = bool(delay.get("is_late_predicted"))
    if delay_risk is None:
        return result

    if not (is_late or float(delay_risk) >= 0.5):
        return result

    market = context.get("market")
    if not market:
        return result

    weather_note = None
    news_note = None
    external_factors = []

    # Weather factor
    try:
        from services.weather_service import fetch_weather

        weather_payload = fetch_weather(region=market)
        weather_items = ((weather_payload or {}).get("weather") or {}).get(market, [])
        risky_weather = [
            item for item in weather_items
            if str(item.get("disruption_risk", "")).lower() in {"high", "medium"}
        ]
        if risky_weather:
            city_bits = [
                f"{item.get('city')} ({item.get('disruption_risk')})"
                for item in risky_weather[:3]
                if item.get("city")
            ]
            if city_bits:
                external_factors.append("weather")
                weather_note = "Weather disruptions are active in key hubs: " + ", ".join(city_bits) + "."
    except Exception:
        pass

    # News / war-conflict factor
    try:
        from services.news_service import fetch_news

        news_payload = fetch_news(market=market)
        articles = (news_payload or {}).get("articles") or []
        war_keywords = {
            "war", "conflict", "attack", "military", "sanction", "sanctions",
            "missile", "airstrike", "red sea", "shipping lane", "blockade",
        }

        war_related = []
        for article in articles:
            title = str(article.get("title", ""))
            title_l = title.lower()
            if any(keyword in title_l for keyword in war_keywords):
                war_related.append(article)

        if war_related:
            top_title = str(war_related[0].get("title", "")).strip()
            external_factors.append("war_news")
            news_note = (
                "Conflict-related news may be affecting routes"
                + (f": '{top_title}'" if top_title else ".")
            )
    except Exception:
        pass

    if not weather_note and not news_note:
        return result

    shipping["external_factors"] = external_factors
    shipping["external_delay_signals"] = {
        "weather": weather_note,
        "news": news_note,
    }
    result["shipping"] = shipping
    return result

# ── Dataset cache (loaded once on first request) ──────────────────────────────
_dataset_cache = None

def _get_dataset():
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    import pandas as pd
    from backend.utils.preprocessing import load_and_clean_csv

    def _iter_csv_paths(base_folder: str):
        folder_path = os.path.join(ROOT, base_folder)
        if not os.path.isdir(folder_path):
            return []
        csv_paths = []
        for dirpath, _, filenames in os.walk(folder_path):
            for fname in sorted(filenames):
                if fname.lower().endswith(".csv"):
                    csv_paths.append(os.path.join(dirpath, fname))
        return sorted(csv_paths)

    expected_cols = [
        "Product_Name",
        "Order_Region",
        "Order_Item_Quantity",
        "Order_Profit_Per_Order",
        "Sales",
        "Category_Name",
        "Market",
        "Order_City",
    ]

    candidates = []
    for folder in ["data/processed", "data/raw"]:
        for csv_path in _iter_csv_paths(folder):
            try:
                df = load_and_clean_csv(csv_path)
            except Exception:
                continue

            col_score = sum(1 for c in expected_cols if c in df.columns)
            name_score = 1 if "supplychain" in os.path.basename(csv_path).lower() else 0
            total_score = (col_score * 10) + name_score
            candidates.append((total_score, len(df), csv_path, df))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_score, _, _, best_df = candidates[0]
        if best_score > 0:
            _dataset_cache = best_df
            return best_df

    raise FileNotFoundError(
        "No compatible CSV found in data/processed/ or data/raw/. "
        "Place the supply chain dataset CSV there (subfolders are supported) and restart."
    )


# ── Dropdown options ──────────────────────────────────────────────────────────

@prediction_bp.route("/options/products", methods=["GET"])
def get_products():
    try:
        df = _get_dataset()
        col = next((c for c in ["Product_Name","Category_Name"] if c in df.columns), None)
        products = sorted(df[col].dropna().unique().tolist()) if col else []
        return jsonify({"products": products})
    except FileNotFoundError as e:
        return jsonify({"products": [], "warning": str(e)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_bp.route("/options/cities", methods=["GET"])
def get_cities():
    try:
        df = _get_dataset()
        col = next((c for c in ["Order_City","Customer_City"] if c in df.columns), None)
        cities = sorted(df[col].dropna().unique().tolist()) if col else []
        return jsonify({"cities": cities})
    except FileNotFoundError as e:
        return jsonify({"cities": [], "warning": str(e)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_bp.route("/options/pincodes", methods=["GET"])
def get_pincodes():
    try:
        df = _get_dataset()
        col = "Customer_Zipcode" if "Customer_Zipcode" in df.columns else None
        if not col:
            return jsonify({"pincodes": []})
        raw = df[col].dropna().unique()
        pincodes = sorted(set(
            str(int(float(p))) for p in raw
            if str(p).strip() not in ("", "nan")
        ))
        return jsonify({"pincodes": pincodes})
    except FileNotFoundError as e:
        return jsonify({"pincodes": [], "warning": str(e)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Producer: demand insight ──────────────────────────────────────────────────

@prediction_bp.route("/producer-overview", methods=["GET"])
def producer_overview():
    try:
        df = _get_dataset()
        from models.producer_models.demand_clustering import predict_overview
        return jsonify(predict_overview(df))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@prediction_bp.route("/demand", methods=["POST"])
def predict_demand():
    body    = request.get_json(silent=True) or {}
    product = body.get("product","").strip()
    city    = body.get("city","").strip()
    if not product or not city:
        return jsonify({"error": "Both 'product' and 'city' are required."}), 400
    try:
        df = _get_dataset()
        from models.producer_models.demand_clustering import predict_for_product_city
        return jsonify(predict_for_product_city(df, product, city))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Consumer: delivery insight ────────────────────────────────────────────────

@prediction_bp.route("/consumer", methods=["POST"])
def predict_consumer():
    body    = request.get_json(silent=True) or {}
    product = body.get("product","").strip()
    pincode = body.get("pincode","").strip()
    if not product or not pincode:
        return jsonify({"error": "Both 'product' and 'pincode' are required."}), 400
    try:
        df = _get_dataset()
        from models.consumer_models.consumer_insights import predict_for_product_pincode
        result = predict_for_product_pincode(df, product, pincode)
        result = _augment_shipping_reason_with_external_signals(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
