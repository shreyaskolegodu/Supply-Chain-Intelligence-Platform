# epoch — Full Project Context Document
# For AI Tool Handoff — Complete Project Understanding

---

## 1. WHAT THIS PROJECT IS

**epoch** is a supply chain intelligence web platform that uses machine learning to give
two types of users actionable insights from a supply chain dataset.

- It is NOT a marketplace. It does NOT let users buy or sell things.
- It is a PREDICTION + INSIGHT tool. Users query the model and get back analytics.
- The ML models are still being built. The codebase uses heuristic stubs right now,
  with clearly marked replacement points for when the real .pkl model files are ready.
- The frontend is served by Flask (not a separate React/Node app).

---

## 2. TWO PORTALS — WHO USES WHAT

### Producer Portal
- Role: A business/seller who wants to understand where demand is for their products.
- Inputs: Product name + Destination City (both from searchable dropdowns).
- Output from model: Demand zone classification (High Demand / Emerging Market / Low Demand),
  demand score (0–100%), total sales, average profit per order, late delivery risk %,
  and a comparison table of the top 5 global regions for that product.

### Consumer Portal
- Role: A customer who wants to know about their delivery before placing/tracking an order.
- Inputs: Product name + Pincode/Zipcode (both from searchable dropdowns).
- Output from model: Delay risk prediction (Safe / Warning / Risk), confidence %,
  estimated delivery days, average delay gap (real vs scheduled), recommended shipping
  mode with a plain-English reason, and location context (market, region, country).

---

## 3. COMPLETE FILE STRUCTURE

```
epoch/
├── frontend/
│   ├── index.html                ← Landing page (portal selector)
│   ├── producer.html             ← Producer dashboard
│   ├── consumer.html             ← Consumer dashboard
│   ├── styles.css                ← Shared dark-theme design system
│   └── app.js                    ← Shared frontend utility functions
│
├── backend/
│   ├── app.py                    ← Flask entry point, serves frontend, registers blueprints
│   ├── routes/
│   │   ├── prediction_routes.py  ← All ML prediction endpoints + dropdown data endpoints
│   │   └── recommendation_routes.py ← Weather + news signal endpoints
│   ├── services/
│   │   ├── weather_service.py    ← OpenWeatherMap API (falls back to mock data)
│   │   └── news_service.py       ← NewsAPI (falls back to mock data)
│   └── utils/
│       ├── preprocessing.py      ← CSV loader, column normaliser, PII stripper
│       └── feature_engineering.py ← Shared feature builders for all models
│
├── models/
│   ├── consumer_models/
│   │   └── consumer_insights.py  ← Unified consumer model (delay + shipping in one file)
│   └── producer_models/
│       └── demand_clustering.py  ← Producer demand zone classifier
│
├── data/
│   ├── raw/                      ← Place original CSV here if not using processed/
│   ├── processed/                ← Preferred location for the dataset CSV
│   └── external/                 ← Weather/news data storage
│
├── notebooks/
│   └── eda.ipynb                 ← EDA notebook (uses data/raw/supply_chain.csv)
│
├── saved_models/                 ← Drop trained .pkl files here when ready
│   ├── cluster_model.pkl         ← (not yet created — stub active)
│   ├── delay_model.pkl           ← (not yet created — stub active)
│   └── risk_model.pkl            ← (not yet created — stub active)
│
├── config/
│   ├── config.yaml               ← API keys + thresholds + model paths
│   └── config.yaml.example       ← Safe template (no secrets)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. DATASET

### Source
Standard e-commerce supply chain dataset. ~180,000 rows. Global orders.

### Key columns used by the models

| Canonical Name (after preprocessing) | Original CSV Name                    | Used By              |
|---------------------------------------|--------------------------------------|----------------------|
| Product_Name                          | Product Name                         | Both portals dropdown|
| Order_City                            | Order City                           | Producer dropdown    |
| Customer_Zipcode                      | Customer Zipcode                     | Consumer dropdown    |
| Order_Region                          | Order Region                         | Both models          |
| Market                                | Market                               | Both models          |
| Sales                                 | Sales                                | Demand clustering    |
| Order_Item_Quantity                   | Order Item Quantity                  | Demand clustering    |
| Order_Profit_Per_Order                | Order Profit Per Order               | Demand + consumer    |
| Late_delivery_risk                    | Late_delivery_risk                   | Consumer insights    |
| Days_for_shipping_real                | Days for shipping (real)             | Consumer insights    |
| Days_for_shipment_scheduled           | Days for shipment (scheduled)        | Consumer insights    |
| Shipping_Mode                         | Shipping Mode                        | Shipping rec.        |
| Customer_City                         | Customer City                        | Context              |
| Customer_Country                      | Customer Country                     | Context              |
| Order_Country                         | Order Country                        | Context              |
| Order_State                           | Order State                          | Context              |
| Order_Status                          | Order Status                         | Context              |
| Category_Name                         | Category Name                        | Fallback for product |

### Markets in the dataset
Africa, Europe, LATAM, Pacific Asia, USCA

### Delivery Status values
Advance shipping, Late delivery, Shipping canceled, Shipping on time

### Order Status values
COMPLETE, PENDING, CLOSED, PENDING_PAYMENT, CANCELED, PROCESSING, SUSPECTED_FRAUD

### Shipping Mode values
Standard Class, First Class, Second Class, Same Day

### PII columns (auto-stripped during preprocessing)
Customer_Email, Customer_Fname, Customer_Lname, Customer_Password,
Customer_Street, Product_Description, Product_Image

---

## 5. HOW THE DATA FLOWS (END TO END)

### Producer flow
1. User opens /producer in browser
2. Frontend calls GET /api/predict/options/products → populates product dropdown
3. Frontend calls GET /api/predict/options/cities   → populates city dropdown
4. User selects product + city, clicks "Get Insights"
5. Frontend sends POST /api/predict/demand with JSON { "product": "...", "city": "..." }
6. Backend loads cached dataset → filters rows matching product + city
7. Calls demand_clustering.predict_for_product_city(df, product, city)
8. Model computes demand score, classifies zone, aggregates stats, finds top 5 regions
9. Returns JSON → frontend renders zone hero card + stats + top regions table

### Consumer flow
1. User opens /consumer in browser
2. Frontend calls GET /api/predict/options/products  → populates product dropdown
3. Frontend calls GET /api/predict/options/pincodes  → populates pincode dropdown
4. User selects product + pincode, clicks "Get Insights"
5. Frontend sends POST /api/predict/consumer with JSON { "product": "...", "pincode": "..." }
6. Backend loads cached dataset → filters by product + pincode
7. If no pincode match → falls back to product-only data (flagged in response)
8. Calls consumer_insights.predict_for_product_pincode(df, product, pincode)
9. Returns delay risk, estimated days, shipping recommendation, location context
10. Frontend renders delay hero card + stats + shipping card + context chips

### Dataset caching
The CSV is loaded once on the first request and cached in memory (_dataset_cache global).
It checks data/processed/ first, then data/raw/. No reload until server restarts.

---

## 6. BACKEND IN DETAIL

### app.py
- Flask app with CORS enabled
- Max upload: 50MB (kept for future use)
- Registers two blueprints: prediction_bp (prefix: /api/predict) and recommendation_bp (prefix: /api/recommend)
- Serves all three HTML pages as static files
- /api/health endpoint returns version from config.yaml

### prediction_routes.py — All active endpoints

GET /api/predict/options/products
  → Returns sorted list of unique Product_Name values from dataset
  → Falls back to Category_Name if Product_Name not found

GET /api/predict/options/cities
  → Returns sorted list of unique Order_City values
  → Falls back to Customer_City

GET /api/predict/options/pincodes
  → Returns sorted list of unique Customer_Zipcode values as clean strings

POST /api/predict/demand
  → Body: { "product": string, "city": string }
  → Calls demand_clustering.predict_for_product_city()
  → Returns: found, product, city, demand_zone, demand_score, market, region,
             total_sales, avg_profit, avg_late_risk, order_count, top_regions[],
             data_source, message

POST /api/predict/consumer
  → Body: { "product": string, "pincode": string }
  → Calls consumer_insights.predict_for_product_pincode()
  → Returns: found, product, pincode, context{}, delay{}, delivery{}, shipping{},
             order_count, avg_sales, data_source

### recommendation_routes.py — Secondary endpoints
GET /api/recommend/weather?region=USCA   → Weather disruption signals per hub city
GET /api/recommend/news?market=Europe    → Supply chain news headlines
(Both fall back to mock data if no API keys configured)

---

## 7. MODELS IN DETAIL

### Producer Model: demand_clustering.py
File: models/producer_models/demand_clustering.py

Main function: predict_for_product_city(df, product, city) → dict

Internal pipeline:
1. Identifies product column (Product_Name or Category_Name)
2. Filters df by product + city
3. Falls back to product-only if no city match
4. Calls build_producer_features() from feature_engineering.py
5. Aggregates: demand_score (mean), total_sales (sum), avg_profit (mean),
   avg_late_risk (mean), order_count
6. Computes zone thresholds: low = 33rd percentile, high = 67th percentile
7. Classifies: score >= high_thresh → "High Demand", >= low_thresh → "Emerging Market", else → "Low Demand"
8. Calls _get_top_regions_for_product() for global context (top 5 regions by score)

Stub function to replace: _classify_demand_zone(score, low_thresh, high_thresh)
Model file to drop: saved_models/cluster_model.pkl

### Consumer Model: consumer_insights.py
File: models/consumer_models/consumer_insights.py

Main function: predict_for_product_pincode(df, product, pincode) → dict

Internal pipeline:
1. Filters df by product
2. Further filters by pincode (Customer_Zipcode) if possible
3. Falls back to product-only subset if no pincode match
4. Runs three sub-predictions:

   _predict_delay_risk(subset) → { delay_risk (0–1), is_late_predicted (bool), confidence (0–100) }
   Currently: uses mean of Late_delivery_risk column
   Replace with: delay_model.pkl

   _predict_estimated_days(subset) → { estimated_days, scheduled_days, avg_gap }
   Currently: means of Days_for_shipping_real and Days_for_shipment_scheduled
   No separate model file — derived from data

   _predict_shipping_mode(subset, delay_risk) → { recommended_mode, reason }
   Currently: rule-based on delay_risk threshold + most common Shipping_Mode
   Replace with: shipping model when ready

   _get_region_context(subset) → { region, market, country }
   Derived from mode of Order_Region, Market, Order_Country columns

### OLD consumer model files (REMOVED — no longer used)
The following files still exist in the repo but are NOT called anywhere:
- models/consumer_models/delay_prediction.py
- models/consumer_models/risk_scoring_model.py
- models/consumer_models/shipping_recommendation.py
These can be deleted or repurposed. All consumer logic now lives in consumer_insights.py.

---

## 8. PREPROCESSING PIPELINE

File: backend/utils/preprocessing.py
Function: load_and_clean_csv(filepath) → pd.DataFrame

Steps:
1. Read CSV with utf-8 encoding, skip bad lines
2. Strip whitespace from all column names
3. Remap column names to canonical names using COLUMN_MAP dict
4. Drop PII columns
5. Cast numeric columns using pd.to_numeric(errors='coerce')
6. Parse date columns: Order_Date, Shipping_Date
7. Drop fully empty rows
8. Return clean DataFrame

Also exports:
- safe_float(series) → numeric series with NaN → 0
- normalise_0_1(series) → min-max normalised series

---

## 9. FEATURE ENGINEERING

File: backend/utils/feature_engineering.py

Functions:
- add_shipping_delay_feature(df) → adds "shipping_delay_gap" = real - scheduled days
- add_profit_margin_feature(df)  → adds "profit_margin" = profit / sales
- add_discount_impact_feature(df)→ adds "high_discount_flag" (1 if discount > 15%)
- add_demand_score(df)           → returns composite score series (Sales 40%, Qty 30%, Profit 20%, Risk_inv 10%)
- add_temporal_features(df)      → adds order_month, order_quarter, order_dow from Order_Date
- build_producer_features(df)    → runs all above for producer model
- build_consumer_features(df)    → runs all above for consumer model

---

## 10. FRONTEND IN DETAIL

### Design system (styles.css)
- Dark theme: bg #08080E, surface #111119, surface2 #191923
- Producer accent colour: --P = #7EB8E8 (electric blue)
- Consumer accent colour: --C = #7AE0B8 (mint green)
- Zone colours: High = #7AE0B8, Emerging = #E8C97A, Low = #E87A7A
- Fonts: Syne (headings, numbers), DM Sans (body), JetBrains Mono (data/code)
- No external CSS framework (Bootstrap, Tailwind etc.) — all custom CSS

### index.html (Landing page)
- Dark grid background
- Two portal cards: Producer (blue) and Consumer (green)
- onclick navigates to /producer or /consumer
- Status pill at bottom showing model state

### producer.html
- Sidebar with nav: Demand Zones (active), Markets, Trends (stubs)
- Searchable product dropdown: calls GET /api/predict/options/products on load
- Searchable city dropdown: calls GET /api/predict/options/cities on load
- Dropdowns filter as user types (up to 80 results shown)
- "Get Insights" button disabled until both values selected
- Results section (hidden until query runs):
  - Warning banner (if no city match found)
  - Zone hero card (class changes: high/emerging/low)
  - Context chips: City, Market, Region, Orders
  - Stats row: Total Sales, Avg Profit/Order, Late Delivery Risk, Historical Orders
  - Top regions table: rank, region name, zone badge, score bar, sales, orders
  - Selected city's region is highlighted in the table

### consumer.html
- Sidebar with nav: Delivery Insights (only one item — no tabs)
- Searchable product dropdown
- Searchable pincode dropdown
- Results section:
  - Delay hero card (class: safe/warning/risk based on delay_risk threshold)
    - safe: delay_risk < 0.4, warning: 0.4–0.6, risk: > 0.6
  - Confidence ring (circular % display)
  - Stats row: Delay Risk %, Est. Delivery Days, Avg Delay Gap, Historical Orders
  - Shipping recommendation card (mode badge + reason text)
  - Context chips: Market, Region, Country, Pincode
  - Data note (explains if pincode match or fallback)

### Searchable dropdown implementation (both pages)
- Input field + hidden field pair per dropdown
- On input: filters allProducts/allCities/allPincodes array, renders matching items
- On item mousedown: sets hidden field value, closes dropdown
- On blur: closes dropdown after 180ms delay (allows click to register first)
- Highlights matching text with <mark> tags
- "Get Insights" button checks both hidden fields are non-empty before enabling

---

## 11. CONFIG (config.yaml)

```yaml
app:
  name: "epoch"
  version: "0.1.0"
  port: 5000
  debug: true

apis:
  openweather_key: ""     # Optional — falls back to mock data
  newsapi_key: ""         # Optional — falls back to mock data

model:
  cluster_model_path:  "saved_models/cluster_model.pkl"
  delay_model_path:    "saved_models/delay_model.pkl"
  risk_model_path:     "saved_models/risk_model.pkl"
  demand_score_weights:
    sales:    0.40
    quantity: 0.30
    profit:   0.20
    risk_inv: 0.10

thresholds:
  high_demand_percentile:    0.67
  low_demand_percentile:     0.33
  risk_critical:             75
  risk_high:                 50
  risk_medium:               25
  delay_gap_warning_days:    2

upload:
  max_size_mb: 50
  allowed_extensions: ["csv"]
```

---

## 12. CURRENT STATUS OF EACH COMPONENT

| Component                              | Status                        | Notes                                      |
|----------------------------------------|-------------------------------|--------------------------------------------|
| Flask backend + routing                | Complete                      | Fully working                              |
| Preprocessing pipeline                 | Complete                      | Handles all column variants                |
| Feature engineering                    | Complete                      | Shared across both models                  |
| Producer demand_clustering.py          | Stub (heuristic)              | Replace _classify_demand_zone() with .pkl  |
| Consumer consumer_insights.py          | Stub (heuristic)              | Replace _predict_delay_risk() with .pkl    |
| Dropdown population (all 3)            | Complete                      | Products, cities, pincodes from dataset    |
| Producer frontend                      | Complete                      | Product + city query, full result render   |
| Consumer frontend                      | Complete                      | Product + pincode query, full result render|
| Landing page                           | Complete                      |                                            |
| Weather service                        | Stub (mock data)              | Add OpenWeatherMap API key to activate     |
| News service                           | Stub (mock data)              | Add NewsAPI key to activate                |
| Markets nav page (producer sidebar)    | Stub (nav item exists, no page) | Not built yet                           |
| Trends nav page (producer sidebar)     | Stub (nav item exists, no page) | Not built yet                           |
| EDA notebook                           | Template ready                | Add your dataset and run cells             |
| saved_models/*.pkl                     | Not yet created               | Models under development                   |

---

## 13. THINGS THAT HAVE BEEN INTENTIONALLY REMOVED

The following were built in earlier versions and then removed by user request:
- CSV file upload on producer dashboard (was: drag-drop zone → bulk demand table)
- CSV file upload on consumer dashboard (was: drag-drop → 3 separate model tabs)
- Delay Prediction tab (separate tab on consumer side)
- Risk Scoring tab (separate tab on consumer side — also had region risk table)
- Shipping Recommendation tab (separate tab on consumer side)
- "I'm here to buy / I'm here to sell" landing page language
- Bulk demand zone table (showing all regions from CSV at once)
- Market filter pills and sort controls on the bulk table

DO NOT bring these back unless the user explicitly asks.

---

## 14. HOW TO PLUG IN TRAINED MODELS

### Step 1: Train and save your model
```python
import joblib
joblib.dump(your_trained_model, "epoch/saved_models/cluster_model.pkl")
```

### Step 2: Producer model replacement
Open: models/producer_models/demand_clustering.py
Find: _classify_demand_zone() function
Replace the stub logic with:
```python
def _classify_demand_zone(score, low_thresh, high_thresh):
    model = joblib.load(os.path.join(ROOT, "saved_models", "cluster_model.pkl"))
    features = [[score]]   # expand to match your model's feature vector
    label_int = model.predict(features)[0]
    zone_map = {2: "High Demand", 1: "Emerging Market", 0: "Low Demand"}
    return zone_map[label_int]
```

### Step 3: Consumer delay model replacement
Open: models/consumer_models/consumer_insights.py
Find: _predict_delay_risk() function
Replace with:
```python
def _predict_delay_risk(subset):
    model  = joblib.load(os.path.join(ROOT, "saved_models", "delay_model.pkl"))
    X      = subset[YOUR_FEATURE_COLS].fillna(0)
    risk   = float(model.predict_proba(X)[:, 1].mean())
    return {
        "delay_risk":        round(risk, 3),
        "is_late_predicted": risk >= 0.5,
        "confidence":        int(abs(risk - 0.5) * 200),
    }
```

### Step 4: No other file changes needed
The frontend, routes, and preprocessing are fully decoupled from the model internals.
The JSON response contract is fixed — as long as your model returns the same dict keys,
everything will render correctly without touching any other file.

---

## 15. HOW TO RUN

```bash
# From project root
pip install -r requirements.txt

# Place dataset CSV in:
#   epoch/data/processed/your_file.csv   ← preferred
#   OR epoch/data/raw/your_file.csv

cd backend
python app.py

# Opens at: http://localhost:5000
# Producer:  http://localhost:5000/producer
# Consumer:  http://localhost:5000/consumer
```

---

## 16. DEPENDENCIES (requirements.txt)

```
flask==3.0.3
flask-cors==4.0.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
joblib==1.4.2
pyyaml==6.0.1
requests==2.32.3
gunicorn==22.0.0
```

---

## 17. CONVENTIONS USED IN THIS CODEBASE

- All model main entry functions are named: run_*() or predict_for_*()
- All stub replacement blocks are marked: # ── REPLACE WITH YOUR MODEL ──
- Column names in Python code use underscore_case (after preprocessing normalisation)
- The frontend uses vanilla JS only — no React, Vue, or jQuery
- CSS variables are defined in :root in styles.css and referenced everywhere
- All API responses include an "error" key when something goes wrong
- Fallback behaviour is always defined — no hard failures on missing data
