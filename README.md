# ⚡ GridSense Labs

**AI-powered real-time energy intelligence for UK homes and businesses.**

> Powered by [Verdant Innovations Ltd](https://github.com/onuigbonnamdi/ai-energy-consumption-optimisation) · Built by Nnamdi Onuigbo

---

## 🌐 Live App

> **[gridsense.streamlit.app](https://gridsense.streamlit.app)** ← *(update with your Streamlit URL after deploying)*

---

## What is GridSense Labs?

GridSense Labs is a freemium AI energy intelligence platform that helps UK households and small businesses understand, predict, and reduce their electricity consumption in real time.

Users enter their last two smart meter readings and instantly receive:

- Their **predicted next 15-minute consumption** powered by a Random Forest AI model
- **Live UK grid price** from the Elexon BMRS API (updated every 5 minutes)
- **Live UK grid demand** as a percentage of national capacity
- **Cost estimates** for the next 15 minutes, today, and this month
- **Carbon footprint** per reading and daily estimate
- **Peak hour warnings** with specific load-shifting advice
- **Appliance cost calculator** — see exactly what each appliance costs to run

Premium tiers (coming soon at gridsense.ai) unlock 24-hour forecasting, 7-day profiles, smart appliance scheduling, and personalised tariff comparison.

---

## Product Tiers

| Feature | Free | Plus £4.99/mo | Pro £9.99/mo |
|---------|------|--------------|-------------|
| Live grid price & demand | ✅ | ✅ | ✅ |
| 15-min AI prediction | ✅ | ✅ | ✅ |
| Normal / Monitor / Reduce recommendation | ✅ | ✅ | ✅ |
| Carbon footprint estimate | ✅ | ✅ | ✅ |
| Peak hour warnings | ✅ | ✅ | ✅ |
| Appliance cost taster | ✅ | ✅ | ✅ |
| 24-hour forecast profile | 🔒 | ✅ | ✅ |
| 7-day forecast | 🔒 | ✅ | ✅ |
| Smart appliance scheduler | 🔒 | ✅ | ✅ |
| Price spike alerts | 🔒 | ✅ | ✅ |
| Postcode regional pricing (DNO-accurate) | 🔒 | ✅ | ✅ |
| Tariff comparison & switching advisor | 🔒 | 🔒 | ✅ |
| Monthly savings report (PDF) | 🔒 | 🔒 | ✅ |
| Multi-site monitoring | 🔒 | 🔒 | ✅ |

---

## How It Works

### 1. Data Layer
- **Elexon BMRS API** — live UK wholesale electricity prices (System Sell/Buy Price) and grid demand, refreshed every 5 minutes
- **Open-Meteo API** — live UK weather (temperature, wind speed, cloud cover), refreshed every 10 minutes
- **Graceful fallback** — if live APIs are unavailable, the app switches to realistic time-aware estimates and clearly labels data as `DEMO MODE`

### 2. AI Model
A **Random Forest Regressor** (120 estimators, max depth 14) trained on 60,000 synthetic household consumption records with the following features:

| Feature | Description |
|---------|-------------|
| Hour of day | 0–23 |
| Month | 1–12 (seasonal patterns) |
| Day of week | 0–6 (weekend uplift) |
| Temperature | °C (heating/cooling demand) |
| Wind speed | m/s (weather-driven demand) |
| Grid price | p/kWh (price-responsive behaviour) |
| Lag 1 | User's reading 15 min ago |
| Lag 2 | User's reading 30 min ago |

Model R² ≥ 0.99 on test set.

### 3. Recommendation Layer
Predicted consumption is mapped to a three-tier decision system:

| Predicted kW | Status | Action |
|-------------|--------|--------|
| < 0.5 kW | ✅ Normal | No action required |
| 0.5 – 1.5 kW | ⚠️ Monitor | Defer non-essential loads |
| > 1.5 kW | 🔴 Reduce Load | Switch off non-essential appliances |

### 4. Cost & Carbon
- **Cost** calculated using the user's entered tariff rate (p/kWh) across 15-min, daily, and monthly windows
- **Carbon** calculated at UK grid average intensity of 233g CO₂/kWh (National Grid ESO)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML Model | scikit-learn RandomForestRegressor |
| Grid Data | Elexon BMRS REST API |
| Weather | Open-Meteo API |
| Data | NumPy, Pandas |
| Visualisation | Matplotlib |
| Deployment | Streamlit Community Cloud |

---

## Repository Structure

```
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── .streamlit/
│   └── config.toml          ← Dark green GridSense theme
└── README.md
```

---

## 🚀 Deploy to Streamlit Community Cloud

1. Fork or clone this repository

2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **New app**

3. Set:
   - **Repository:** `onuigbonnamdi/gridsense-labs`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - Click **Deploy**

4. App goes live in ~2 minutes ✅

### Run Locally

```bash
git clone https://github.com/onuigbonnamdi/gridsense-labs
cd gridsense-labs
pip install -r requirements.txt
streamlit run app.py
```

---

## Relationship to Verdant Innovations

GridSense Labs is the **consumer and SME-facing product** built on the same energy intelligence infrastructure developed by Verdant Innovations Ltd.

| | Verdant Innovations | GridSense Labs |
|-|-------------------|----------------|
| **Audience** | Grid operators, large enterprises | UK households, SMEs |
| **Data** | Elexon BMRS, Open-Meteo, Prophet forecasting | Elexon BMRS, Open-Meteo |
| **Product** | Energy Insight Reports, API access, B2B analytics | Freemium consumer app |
| **Pricing** | Enterprise | Free → £9.99/month |

GridSense Labs acts as the **top of the funnel** — users with complex multi-site or commercial needs are directed to Verdant Innovations for enterprise solutions.

---

## Roadmap

- [ ] Stripe payment integration for Plus and Pro tiers
- [ ] User accounts and usage history
- [ ] Push / email price spike alerts
- [ ] 24-hour and 7-day forecast (Prophet model)
- [ ] Smart appliance scheduling engine
- [ ] **Postcode-level regional pricing** — DNO region detection, region-accurate unit rates and standing charges across all 14 UK DNO regions
- [ ] Tariff comparison with live Ofgem data and postcode-adjusted rates
- [ ] Mobile app (React Native)
- [ ] gridsense.ai domain launch

---

## Data Sources & Attribution

- **Elexon BMRS API** — [data.elexon.co.uk](https://data.elexon.co.uk) · Open Government Licence
- **Open-Meteo** — [open-meteo.com](https://open-meteo.com) · CC BY 4.0
- **UK Grid Carbon Intensity** — National Grid ESO, 233g CO₂/kWh (2024 average)
- **Tariff data** — indicative averages, not financial advice

---

## Author

**Nnamdi Onuigbo**
AI Systems Engineer | Founder, Verdant Innovations Ltd & GridSense Labs
[GitHub](https://github.com/onuigbonnamdi) · [SmartFlow Systems](https://smartflowsys.com)

---

*GridSense Labs is an independent product. Energy cost estimates are for informational purposes only and do not constitute financial or energy advice.*
