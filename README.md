# epoch — Supply Chain Intelligence Platform

ML-powered supply chain analytics with demand zone insights for producers and delivery intelligence for consumers.

---

## Project Structure

```
epoch/
├── frontend/                     → HTML/CSS/JS dashboard (served by Flask)
│   ├── index.html                → Landing page
│   ├── producer.html             → Producer dashboard
│   ├── consumer.html             → Consumer dashboard
│   ├── styles.css
│   └── app.js
│
├── backend/                      → Flask API server
│   ├── app.py
│   ├── routes/
│   │   ├── prediction_routes.py  → All prediction + dropdown endpoints
│   │   └── recommendation_routes.py → Weather + news signals
│   ├── services/
│   │   ├── weather_service.py
│   │   └── news_service.py
│   └── utils/
│       ├── preprocessing.py
│       └── feature_engineering.py
│
├── models/
│   ├── consumer_models/
│   │   └── consumer_insights.py  → Unified: delay risk + shipping recommendation
│   └── producer_models/
│       └── demand_clustering.py  → Demand zone classifier
│
├── data/
│   ├── raw/                      → Original dataset (place CSV here)
│   ├── processed/                → Cleaned dataset (preferred location)
│   └── external/                 → Weather / news data
│
├── notebooks/
│   └── eda.ipynb                 → EDA and experimentation
│
├── saved_models/                 → Trained models (.pkl files go here)
│   ├── delay_model.pkl
│   ├── risk_model.pkl
│   └── cluster_model.pkl
│
├── config/
│   └── config.yaml               → API keys + model parameters
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
cd epoch
pip install -r requirements.txt
```

### 2. Place your dataset
Drop your supply chain CSV into `data/processed/` (preferred) or `data/raw/`. The backend auto-detects and loads it on first request — no configuration needed.

### 3. Configure API keys (optional)
Edit `config/config.yaml`:
```yaml
apis:
  openweather_key: "YOUR_KEY_HERE"
  newsapi_key:     "YOUR_KEY_HERE"
```
Without keys, weather and news services fall back to realistic mock data automatically.

### 4. Run the backend
```bash
cd backend
python app.py
```
Open http://localhost:5000

---

## How It Works

There are no file uploads on either portal. Both dashboards use searchable dropdowns populated directly from your dataset. The user selects inputs and the model returns insights instantly.

### Producer Dashboard
**Inputs:** Product + Destination City
**Output:** Demand zone (High / Emerging / Low), demand score, total sales, average profit, late delivery risk, and a global top-regions comparison table for that product.

### Consumer Dashboard
**Inputs:** Product + Pincode
**Output:** Delay risk prediction, confidence score, estimated delivery days, average delay gap, and recommended shipping mode with reasoning.

---

## API Endpoints

| Method | Endpoint                          | Description                                  |
|--------|-----------------------------------|----------------------------------------------|
| GET    | `/api/predict/options/products`   | Unique product names from dataset            |
| GET    | `/api/predict/options/cities`     | Unique destination cities from dataset       |
| GET    | `/api/predict/options/pincodes`   | Unique customer zipcodes from dataset        |
| POST   | `/api/predict/demand`             | Producer: product + city → demand insights   |
| POST   | `/api/predict/consumer`           | Consumer: product + pincode → delivery insights |
| GET    | `/api/recommend/weather`          | Weather signals by region                    |
| GET    | `/api/recommend/news`             | News signals by market                       |
| GET    | `/api/health`                     | Health check                                 |

### Request format for prediction endpoints

**Producer**
```json
POST /api/predict/demand
{ "product": "Field & Stream Sportsman 16 Gun Fire Safe", "city": "Chicago" }
```

**Consumer**
```json
POST /api/predict/consumer
{ "product": "Field & Stream Sportsman 16 Gun Fire Safe", "pincode": "95758" }
```

---

## Plugging In Your Trained Models

Each model file has a clearly marked `── REPLACE WITH YOUR MODEL ──` block. Drop your `.pkl` files into `saved_models/` and uncomment the loader lines — no other changes needed.

### Producer: Demand Clustering
File: `models/producer_models/demand_clustering.py`
```python
# Replace _classify_demand_zone() with:
model     = joblib.load("saved_models/cluster_model.pkl")
features  = [[demand_score, avg_profit, avg_late_risk, ...]]
label_int = model.predict(features)[0]
zone_map  = {2: "High Demand", 1: "Emerging Market", 0: "Low Demand"}
return zone_map[label_int]
```

### Consumer: Delay Prediction
File: `models/consumer_models/consumer_insights.py`
```python
# Replace _predict_delay_risk() with:
model  = joblib.load("saved_models/delay_model.pkl")
X      = subset[FEATURE_COLS]
risk   = float(model.predict_proba(X)[:, 1].mean())
is_late = risk >= 0.5
```

### Consumer: Shipping Recommendation
File: `models/consumer_models/consumer_insights.py`
```python
# Replace _predict_shipping_mode() with:
model = joblib.load("saved_models/shipping_model.pkl")
X     = subset[FEATURE_COLS]
mode  = model.predict(X)[0]
```

---

## Dataset Columns Used

| Column                          | Used by                        |
|---------------------------------|--------------------------------|
| `Product Name`                  | Both portals (dropdown)        |
| `Order City`                    | Producer portal (dropdown)     |
| `Customer Zipcode`              | Consumer portal (dropdown)     |
| `Order Region`                  | Both models                    |
| `Market`                        | Both models                    |
| `Sales`                         | Demand clustering              |
| `Order Item Quantity`           | Demand clustering              |
| `Order Profit Per Order`        | Demand clustering + insights   |
| `Late_delivery_risk`            | Consumer insights              |
| `Days for shipping (real)`      | Consumer insights              |
| `Days for shipment (scheduled)` | Consumer insights              |
| `Shipping Mode`                 | Shipping recommendation        |

Column names are normalised automatically — spacing and capitalisation variations are handled by `preprocessing.py`.

---

## Development Notes

- The dataset is loaded once on first request and cached in memory — no repeated disk reads.
- If no pincode match is found, the consumer model falls back to product-level historical data and flags this in the response.
- If no city match is found, the producer model falls back to product-level data and shows a warning banner in the UI.
- PII columns (email, name, password, street) are stripped automatically during preprocessing.
- Feature engineering is shared via `backend/utils/feature_engineering.py` — add new features here so both models benefit.
