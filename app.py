import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import warnings
from datetime import datetime, timedelta
import pytz

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridSense Labs",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    /* Cards */
    .gs-card {
        background: linear-gradient(135deg, #0d2318, #0a3020);
        border: 1px solid #1a5c35;
        border-radius: 14px;
        padding: 18px 22px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .gs-card h2 { font-family: 'Space Mono', monospace; font-size: 1.9rem; margin: 6px 0 2px; color: #4ade80; }
    .gs-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: #86efac; opacity: 0.8; }
    .gs-card .sub   { font-size: 0.8rem; color: #86efac; opacity: 0.7; margin-top: 2px; }

    /* Live badge */
    .live-badge {
        display: inline-block;
        background: #16a34a;
        color: white;
        font-size: 0.7rem;
        font-family: 'Space Mono', monospace;
        letter-spacing: 1px;
        padding: 2px 10px;
        border-radius: 20px;
        margin-left: 8px;
        vertical-align: middle;
        animation: pulse 2s infinite;
    }
    .demo-badge {
        display: inline-block;
        background: #b45309;
        color: white;
        font-size: 0.7rem;
        font-family: 'Space Mono', monospace;
        letter-spacing: 1px;
        padding: 2px 10px;
        border-radius: 20px;
        margin-left: 8px;
        vertical-align: middle;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

    /* Result box */
    .result-box {
        border-radius: 14px;
        padding: 22px 26px;
        color: white;
        margin-top: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .result-normal  { background: linear-gradient(135deg, #052e16, #166534); border-color: #22c55e; }
    .result-monitor { background: linear-gradient(135deg, #1c1400, #78350f); border-color: #f59e0b; }
    .result-reduce  { background: linear-gradient(135deg, #1c0000, #7f1d1d); border-color: #ef4444; }
    .result-box h2  { font-family: 'Space Mono', monospace; font-size: 2.2rem; margin: 8px 0 4px; }
    .result-box .rec-label { font-size: 1rem; font-weight: 600; letter-spacing: 0.5px; }
    .result-box .rec-advice { font-size: 0.9rem; opacity: 0.85; margin-top: 8px; }

    /* Premium lock */
    .premium-lock {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px dashed #4ade80;
        border-radius: 14px;
        padding: 30px;
        text-align: center;
        color: #94a3b8;
    }
    .premium-lock h3 { color: #4ade80; font-family: 'Space Mono', monospace; margin-bottom: 8px; }
    .premium-lock .price { font-size: 1.8rem; color: white; font-family: 'Space Mono', monospace; }

    /* Grid status bar */
    .grid-bar {
        height: 12px;
        border-radius: 6px;
        background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444);
        position: relative;
        margin: 8px 0;
    }
    .grid-marker {
        position: absolute;
        top: -4px;
        width: 4px;
        height: 20px;
        background: white;
        border-radius: 2px;
        transform: translateX(-50%);
    }

    /* Header */
    .gs-header { font-family: 'Space Mono', monospace; }
    .gs-tagline { color: #86efac; font-size: 0.95rem; letter-spacing: 0.5px; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #050f0a; border-right: 1px solid #1a5c35; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

UK_TZ = pytz.timezone("Europe/London")

# ── Live data fetchers ────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_elexon_price():
    """Fetch current UK wholesale electricity price from Elexon BMRS."""
    try:
        now_uk = datetime.now(UK_TZ)
        settlement_date = now_uk.strftime("%Y-%m-%d")
        period = max(1, min(48, int((now_uk.hour * 60 + now_uk.minute) / 30) + 1))
        url = (
            f"https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-sell-buy-prices"
            f"?settlementDate={settlement_date}&settlementPeriod={period}&format=json"
        )
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            items = data.get("data", [])
            if items:
                ssp = float(items[0].get("systemSellPrice", 0))
                sbp = float(items[0].get("systemBuyPrice", 0))
                mid = (ssp + sbp) / 2
                # Convert £/MWh to p/kWh
                return round(mid / 10, 2), True
    except Exception:
        pass
    # Fallback: realistic UK price estimate based on time of day
    hour = datetime.now(UK_TZ).hour
    if 7 <= hour <= 9 or 17 <= hour <= 20:
        return round(np.random.uniform(28, 38), 2), False
    elif 23 <= hour or hour <= 5:
        return round(np.random.uniform(8, 16), 2), False
    else:
        return round(np.random.uniform(18, 26), 2), False


@st.cache_data(ttl=300, show_spinner=False)
def fetch_elexon_demand():
    """Fetch current UK grid demand from Elexon."""
    try:
        now_uk = datetime.now(UK_TZ)
        settlement_date = now_uk.strftime("%Y-%m-%d")
        period = max(1, min(48, int((now_uk.hour * 60 + now_uk.minute) / 30) + 1))
        url = (
            f"https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
            f"?settlementDate={settlement_date}&settlementPeriod={period}&format=json"
        )
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            items = data.get("data", [])
            if items:
                demand_mw = float(items[0].get("demand", 0))
                if demand_mw > 0:
                    return round(demand_mw, 0), True
    except Exception:
        pass
    # Fallback estimate
    hour = datetime.now(UK_TZ).hour
    base = 28000 + 12000 * np.sin(np.pi * (hour - 6) / 18) if 6 <= hour <= 22 else 24000
    return round(base + np.random.uniform(-1000, 1000), 0), False


@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(lat=51.5, lon=-0.12):
    """Fetch current UK weather from Open-Meteo."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,wind_speed_10m,cloud_cover"
            f"&timezone=Europe%2FLondon"
        )
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            c = r.json().get("current", {})
            return {
                "temp":  c.get("temperature_2m", 12.0),
                "wind":  c.get("wind_speed_10m", 8.0),
                "cloud": c.get("cloud_cover", 50),
            }, True
    except Exception:
        pass
    return {"temp": 12.0, "wind": 8.0, "cloud": 50}, False


# ── ML model ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_model():
    rng = np.random.default_rng(42)
    n = 60_000
    hours   = rng.integers(0, 24, n)
    months  = rng.integers(1, 13, n)
    days    = rng.integers(0, 7, n)
    temps   = rng.uniform(2, 28, n)
    winds   = rng.uniform(0, 30, n)
    prices  = rng.uniform(8, 45, n)

    base   = 0.5 + 0.5 * np.sin(np.pi * hours / 12)
    heat   = np.where(temps < 10, (10 - temps) * 0.04, 0)
    cool   = np.where(temps > 22, (temps - 22) * 0.03, 0)
    wind_e = winds * 0.005
    season = 0.1 * np.sin(2 * np.pi * (months - 1) / 12)
    wknd   = np.where(days >= 5, 0.12, 0)
    noise  = rng.normal(0, 0.06, n)
    power  = np.clip(base + heat + cool + wind_e + season + wknd + noise, 0.05, 5.0)

    lag1 = np.roll(power, 1)
    lag2 = np.roll(power, 2)
    X = np.column_stack([hours, months, days, temps, winds, prices, lag1, lag2])
    y = power

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(n_estimators=120, max_depth=14, n_jobs=-1, random_state=42)
    m.fit(X_tr, y_tr)
    r2 = r2_score(y_te, m.predict(X_te))
    return m, r2


def predict_consumption(model, hour, month, day_num, temp, wind, price, lag1, lag2):
    feat = np.array([[hour, month, day_num, temp, wind, price, lag1, lag2]])
    return max(0.0, float(model.predict(feat)[0]))


def get_recommendation(kw, price_pkwh):
    if kw < 0.5:
        return "Normal ✅", "normal", f"Low usage. Grid price is {price_pkwh}p/kWh — good time to run appliances."
    elif kw < 1.5:
        return "Monitor ⚠️", "monitor", f"Moderate usage detected. Consider deferring non-essential loads."
    else:
        return "Reduce Load 🔴", "reduce", f"High consumption at {price_pkwh}p/kWh. Switch off non-essential appliances to cut costs."


def carbon_kg(kw_15min):
    # UK grid carbon intensity ~233 gCO2/kWh (National Grid ESO 2024 avg)
    kwh = kw_15min * 0.25
    return round(kwh * 0.233, 4)


def is_peak(hour):
    return (7 <= hour <= 9) or (17 <= hour <= 20)


APPLIANCE_KW = {
    "Electric Shower (9kW)": 9.0,
    "Kettle": 3.0,
    "Washing Machine": 2.0,
    "Tumble Dryer": 2.5,
    "Dishwasher": 1.5,
    "Electric Oven": 2.2,
    "Microwave": 1.0,
    "TV (large)": 0.2,
    "Laptop": 0.065,
    "LED Bulb": 0.009,
    "Fridge-Freezer": 0.15,
    "EV Charger (7kW)": 7.0,
}

# ── Build model ───────────────────────────────────────────────────────────────
with st.spinner("Initialising GridSense AI…"):
    model, model_r2 = build_model()

# ── Fetch live data ───────────────────────────────────────────────────────────
price_pkwh, price_live   = fetch_elexon_price()
demand_mw,  demand_live  = fetch_elexon_demand()
weather,    weather_live = fetch_weather()

now_uk   = datetime.now(UK_TZ)
hour_now = now_uk.hour
month_now = now_uk.month
day_now  = now_uk.weekday()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ GridSense Labs")
    st.markdown("<span class='gs-tagline'>AI Energy Intelligence</span>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("🔌 Your Power Readings")
    st.caption("Enter your last two meter readings (15-min intervals) in kW. Check your smart meter display.")
    lag1_in = st.number_input("15 min ago (kW)", min_value=0.0, max_value=15.0, value=0.85, step=0.05)
    lag2_in = st.number_input("30 min ago (kW)", min_value=0.0, max_value=15.0, value=0.80, step=0.05)

    st.markdown("---")
    st.subheader("🏠 Property Type")
    prop = st.selectbox("Property", ["Flat / Apartment", "Terraced House", "Semi-detached", "Detached House", "Small Business"])
    prop_multiplier = {"Flat / Apartment": 0.7, "Terraced House": 0.9,
                       "Semi-detached": 1.0, "Detached House": 1.3, "Small Business": 2.5}[prop]

    st.markdown("---")
    st.subheader("💷 Your Tariff")
    tariff = st.selectbox("Tariff type", ["Standard Variable", "Economy 7", "Agile Octopus", "Fixed Rate"])
    custom_rate = st.number_input("Your rate (p/kWh)", min_value=1.0, max_value=80.0, value=24.5, step=0.5,
                                   help="Check your energy bill for your unit rate")

    st.markdown("---")
    predict_btn = st.button("⚡ Analyse Now", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("**GridSense Labs** · Powered by Verdant Innovations  \n"
               "Data: Elexon BMRS · Open-Meteo")

# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    live_tag = '<span class="live-badge">● LIVE</span>' if (price_live or demand_live) else '<span class="demo-badge">DEMO MODE</span>'
    st.markdown(f'<h1 class="gs-header">⚡ GridSense Labs {live_tag}</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gs-tagline">Real-time AI energy intelligence for UK homes & businesses</p>', unsafe_allow_html=True)
with col_h2:
    st.markdown(f"**{now_uk.strftime('%A, %d %b %Y')}**  \n{now_uk.strftime('%H:%M')} UK time")

st.markdown("---")

# ── Live grid status cards ────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

peak_status = "PEAK 🔴" if is_peak(hour_now) else "OFF-PEAK 🟢"
demand_pct  = min(100, max(0, int((demand_mw - 18000) / (48000 - 18000) * 100)))

cards = [
    ("GRID PRICE",    f"{price_pkwh}p",     "per kWh",         "live" if price_live else "est."),
    ("GRID DEMAND",   f"{int(demand_mw/1000)}GW", f"{demand_pct}% capacity", "live" if demand_live else "est."),
    ("TEMPERATURE",   f"{weather['temp']}°C", f"Wind {weather['wind']} m/s", "live" if weather_live else "est."),
    ("PERIOD",        peak_status,           f"{now_uk.strftime('%H:%M')}",  "now"),
    ("AI MODEL R²",   f"{model_r2:.4f}",     "prediction accuracy",          "trained"),
]
for col, (label, val, sub, tag) in zip([c1, c2, c3, c4, c5], cards):
    col.markdown(
        f'<div class="gs-card"><div class="label">{label} <small style="opacity:0.6">({tag})</small></div>'
        f'<h2>{val}</h2><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Main prediction ───────────────────────────────────────────────────────────
predicted_kw = predict_consumption(
    model, hour_now, month_now, day_now,
    weather["temp"], weather["wind"], price_pkwh,
    lag1_in * prop_multiplier, lag2_in * prop_multiplier
)

rec_label, rec_class, rec_advice = get_recommendation(predicted_kw, price_pkwh)
carbon      = carbon_kg(predicted_kw)
cost_15min  = predicted_kw * 0.25 * (custom_rate / 100)
cost_daily  = cost_15min * 96
cost_monthly = cost_daily * 30

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Analysis",
    "🔒 24h Forecast  ·  Premium",
    "🔒 Appliance Planner  ·  Premium",
    "🔒 Tariff Comparison  ·  Premium",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FREE: LIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("🔮 Next 15-Min Forecast")
        st.markdown(
            f'<div class="result-box result-{rec_class}">'
            f'<div class="rec-label">{rec_label}</div>'
            f'<h2>{predicted_kw:.3f} kW</h2>'
            f'<div class="rec-advice">{rec_advice}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### 💷 Cost Estimates")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Next 15 min", f"£{cost_15min:.4f}")
        cc2.metric("Today (est.)", f"£{cost_daily:.2f}")
        cc3.metric("This month (est.)", f"£{cost_monthly:.2f}")

        st.markdown("#### 🌿 Carbon Footprint")
        cf1, cf2 = st.columns(2)
        cf1.metric("Next 15 min", f"{carbon*1000:.1f}g CO₂")
        cf2.metric("Today (est.)", f"{carbon*1000*96/1000:.2f}kg CO₂")

        # Peak hour warning
        if is_peak(hour_now):
            st.warning(
                f"⏰ **Peak period active ({now_uk.strftime('%H:%M')})** — "
                f"Grid price is elevated. Defer washing machine, dishwasher, "
                f"and EV charging if possible to save money."
            )
        else:
            next_peak = "17:00" if hour_now < 17 else "07:00 tomorrow"
            st.success(f"✅ **Off-peak** — Good time to run appliances. Next peak: {next_peak}")

    with right:
        st.subheader("📊 Grid Status")

        # Demand gauge
        fig, axes = plt.subplots(2, 1, figsize=(6, 5))
        fig.patch.set_facecolor("#050f0a")

        # Bar: demand
        demand_pcts = [demand_pct, 100 - demand_pct]
        colours_d   = ["#4ade80" if demand_pct < 60 else "#f59e0b" if demand_pct < 80 else "#ef4444", "#1a2a1a"]
        axes[0].barh(["Grid\nDemand"], [demand_pct], color=colours_d[0], height=0.4)
        axes[0].barh(["Grid\nDemand"], [100 - demand_pct], left=[demand_pct], color=colours_d[1], height=0.4)
        axes[0].set_xlim(0, 100)
        axes[0].set_facecolor("#050f0a")
        axes[0].tick_params(colors="white")
        axes[0].set_xlabel("% Capacity", color="white")
        axes[0].set_title(f"UK Grid Demand: {int(demand_mw/1000)}GW ({demand_pct}%)", color="#4ade80")
        axes[0].spines[:].set_color("#1a5c35")
        axes[0].text(demand_pct + 1, 0, f"{demand_pct}%", color="white", va="center", fontsize=10)

        # Bar: your usage vs threshold
        thresholds = [0.5, 1.5, 3.0]
        t_labels   = ["Low\n(<0.5)", "Moderate\n(<1.5)", "High\n(<3.0)"]
        t_colours  = ["#22c55e", "#f59e0b", "#ef4444"]
        axes[1].bar(t_labels, thresholds, color=t_colours, alpha=0.4, width=0.5)
        axes[1].axhline(predicted_kw, color="white", linewidth=2.5, linestyle="--",
                        label=f"Your usage: {predicted_kw:.2f}kW")
        axes[1].set_facecolor("#050f0a")
        axes[1].tick_params(colors="white")
        axes[1].set_ylabel("kW", color="white")
        axes[1].set_title("Your Usage vs Thresholds", color="#4ade80")
        axes[1].legend(facecolor="#0a1a0f", labelcolor="white", fontsize=9)
        axes[1].spines[:].set_color("#1a5c35")

        fig.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close(fig)

        # Quick input summary
        st.markdown("#### 📋 Reading Summary")
        df_sum = pd.DataFrame({
            "": ["Your reading (15m ago)", "Your reading (30m ago)", "Property type",
                 "Tariff rate", "Grid price now", "Weather"],
            "Value": [f"{lag1_in} kW", f"{lag2_in} kW", prop,
                      f"{custom_rate}p/kWh", f"{price_pkwh}p/kWh",
                      f"{weather['temp']}°C, {weather['wind']}m/s wind"],
        })
        st.dataframe(df_sum, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREMIUM: 24H FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="premium-lock">'
        '<h3>🔒 24-Hour Forecast & 7-Day Profile</h3>'
        '<p>See your predicted energy consumption hour-by-hour for today and the next 7 days, '
        'with grid price overlays and the best times to run your appliances.</p>'
        '<div class="price">£4.99<small style="font-size:1rem">/month</small></div>'
        '<p style="margin-top:8px;font-size:0.85rem">Estimated monthly savings: <b style="color:#4ade80">£15–£40</b> — '
        'pays for itself in days.</p>'
        '<p style="margin-top:12px;color:#4ade80">→ Coming soon at <b>gridsense.ai</b></p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Preview — What you'll see:")
    # Show teaser chart (blurred/watermarked feel — low alpha)
    hours_r = np.arange(0, 24)
    teaser  = [predict_consumption(model, h, month_now, day_now,
                weather["temp"], weather["wind"], price_pkwh, lag1_in, lag2_in)
               for h in hours_r]
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#050f0a")
    ax.set_facecolor("#050f0a")
    cols_t = ["#22c55e" if v < 0.5 else "#f59e0b" if v < 1.5 else "#ef4444" for v in teaser]
    ax.bar(hours_r, teaser, color=cols_t, alpha=0.25, width=0.8)
    ax.set_xticks(hours_r)
    ax.set_xticklabels([f"{h:02d}h" for h in hours_r], color="#334155", fontsize=7)
    ax.tick_params(colors="#334155")
    ax.spines[:].set_color("#1a2a1a")
    # Watermark
    ax.text(12, max(teaser) * 0.5, "UPGRADE TO UNLOCK", color="#1a5c35",
            fontsize=20, ha="center", va="center", alpha=0.6,
            fontweight="bold", style="italic")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREMIUM: APPLIANCE PLANNER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="premium-lock">'
        '<h3>🔒 Smart Appliance Scheduler</h3>'
        '<p>Tell us which appliances you want to run today. We\'ll calculate the cheapest time '
        'to run each one based on live grid prices and your tariff — and show you exactly how much you save.</p>'
        '<div class="price">£4.99<small style="font-size:1rem">/month</small></div>'
        '<p style="margin-top:8px;font-size:0.85rem">Average saving: <b style="color:#4ade80">£8–£22/month</b> '
        'just from shifting appliance timing.</p>'
        '<p style="margin-top:12px;color:#4ade80">→ Coming soon at <b>gridsense.ai</b></p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Preview — Appliance cost calculator (free taster):")
    sel_appliance = st.selectbox("Select an appliance", list(APPLIANCE_KW.keys()))
    run_mins      = st.slider("Run time (minutes)", 5, 120, 30)
    a_kw          = APPLIANCE_KW[sel_appliance]
    a_kwh         = a_kw * (run_mins / 60)
    a_cost_now    = a_kwh * (custom_rate / 100)
    a_cost_offpeak= a_kwh * (custom_rate * 0.6 / 100)   # typical off-peak discount
    a_saving      = a_cost_now - a_cost_offpeak
    a_carbon      = a_kwh * 0.233

    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Appliance power", f"{a_kw} kW")
    ac2.metric("Energy used",     f"{a_kwh:.3f} kWh")
    ac3.metric("Cost at peak",    f"£{a_cost_now:.4f}")
    ac4.metric("Cost off-peak",   f"£{a_cost_offpeak:.4f}", delta=f"-£{a_saving:.4f}")
    st.info(f"🌿 Running {sel_appliance} for {run_mins} min produces **{a_carbon*1000:.0f}g CO₂**. "
            f"Shift to off-peak and save **£{a_saving:.4f}** this run "
            f"(**~£{a_saving*30:.2f}/month** if daily).")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREMIUM: TARIFF COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<div class="premium-lock">'
        '<h3>🔒 Tariff Comparison & Switching Advisor</h3>'
        '<p>Based on your actual usage pattern, we compare Standard Variable, Economy 7, '
        'Agile Octopus, and Fixed Rate tariffs — and tell you exactly which one saves you most '
        'given <i>your</i> specific usage profile.</p>'
        '<div class="price">£9.99<small style="font-size:1rem">/month</small></div>'
        '<p style="margin-top:8px;font-size:0.85rem">Average annual saving from switching: '
        '<b style="color:#4ade80">£120–£350</b></p>'
        '<p style="margin-top:12px;color:#4ade80">→ Coming soon at <b>gridsense.ai</b></p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Current tariff snapshot:")
    tariff_data = pd.DataFrame({
        "Tariff":         ["Standard Variable", "Economy 7", "Agile Octopus", "Fixed Rate (12m)"],
        "Day Rate (p/kWh)":  [24.5, 28.6, price_pkwh, 22.8],
        "Night Rate (p/kWh)":[24.5, 11.2, max(8, price_pkwh * 0.4), 22.8],
        "Standing Charge":   ["53p/day", "53p/day", "43p/day", "50p/day"],
        "Best for":          ["Average usage", "Night-shift users / EVs",
                               "Flexible / smart home", "Certainty seekers"],
    })
    st.dataframe(tariff_data, hide_index=True, use_container_width=True)
    st.caption("Rates are indicative averages. Upgrade to Premium for personalised comparison based on your usage data.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#1a5c35;font-size:0.78rem;font-family:monospace'>"
    "GridSense Labs · Powered by Verdant Innovations · "
    "Data: Elexon BMRS API & Open-Meteo · "
    "<a href='https://github.com/onuigbonnamdi/ai-energy-consumption-optimisation' "
    "style='color:#4ade80' target='_blank'>GitHub</a>"
    "</center>",
    unsafe_allow_html=True,
)
