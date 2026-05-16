"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        TELENOR NETWORK CAPEX & COVERAGE OPTIMIZER                           ║
║        Principal Full-Stack Data Science & Telecom Enterprise Architecture  ║
║        Region Focus: Hyderabad & Rural Sindh, Pakistan                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Author        : Lokesh Kumar
Role          : Lead Telecom Data Scientist
Stack         : Streamlit · Folium · Plotly · Pandas · NumPy
Last Updated  : 2026
Description   : A production-grade interactive CAPEX planning and geospatial
                coverage optimization tool for Telenor's Sindh network rollout.
                All data is synthetically generated; no external DB required.
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import io
import math
import warnings
warnings.filterwarnings("ignore")

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ══════════════════════════════════════════════════════════════════════════════
# 0.  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Telenor | CAPEX & Coverage Optimizer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  GLOBAL CSS  –  Glassmorphism / Telenor Corporate Dark Theme
# ══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Palette ── */
:root {
    --navy:       #050D1F;
    --navy-mid:   #0A1931;
    --navy-card:  #0D2240;
    --slate:      #1C3A6B;
    --electric:   #00A8FF;
    --electric-2: #0070CC;
    --accent:     #00E5FF;
    --warn:       #FF6B35;
    --ok:         #00D68F;
    --text-h:     #E8F4FF;
    --text-b:     #8BACC8;
    --glass:      rgba(13, 34, 64, 0.72);
    --border:     rgba(0, 168, 255, 0.18);
    --glow:       0 0 24px rgba(0,168,255,0.22);
}

/* ── Base App Background ── */
.stApp {
    background: linear-gradient(135deg, #050D1F 0%, #060f22 40%, #091628 100%);
    font-family: 'DM Sans', sans-serif;
    color: var(--text-b);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060e1e 0%, #091628 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-b) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text-h) !important; }

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-h) !important;
    letter-spacing: -0.5px;
}

/* ── KPI Card ── */
.kpi-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 22px;
    backdrop-filter: blur(14px);
    box-shadow: var(--glow), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 4px;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 36px rgba(0,168,255,0.32), inset 0 1px 0 rgba(255,255,255,0.06);
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--text-b);
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: var(--electric);
    line-height: 1.1;
}
.kpi-sub {
    font-size: 12px;
    color: var(--text-b);
    margin-top: 4px;
}
.kpi-delta-up   { color: var(--ok);   font-size: 12px; font-weight: 600; }
.kpi-delta-down { color: var(--warn); font-size: 12px; font-weight: 600; }

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-bottom: 20px;
}
.section-header h2 {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text-h) !important;
    margin: 0 !important;
}
.section-pill {
    background: var(--electric);
    color: var(--navy) !important;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
}

/* ── Glass Panel ── */
.glass-panel {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    backdrop-filter: blur(12px);
    margin-bottom: 18px;
}

/* ── Dataframe Theming ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--electric-2), var(--electric)) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 20px rgba(0,168,255,0.35) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(0,168,255,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] .st-be { background: var(--electric) !important; }
[data-testid="stSlider"] .st-bf { background: var(--slate) !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div {
    background: var(--navy-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-h) !important;
}

/* ── Success / Info / Warning boxes ── */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--glass) !important;
}

/* ── Logo header ── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 24px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.brand-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: var(--text-h);
    letter-spacing: 2px;
    text-transform: uppercase;
    line-height: 1.4;
}
.brand-sub {
    font-size: 11px;
    color: var(--text-b);
    letter-spacing: 0.5px;
}
.brand-dot {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, var(--electric-2), var(--accent));
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 18px rgba(0,168,255,0.45);
    flex-shrink: 0;
}

/* ── Priority Badge ── */
.badge-critical { color: #FF4D4D; font-weight: 700; }
.badge-high     { color: #FF9500; font-weight: 700; }
.badge-medium   { color: var(--electric); font-weight: 700; }
.badge-low      { color: var(--ok); font-weight: 700; }

/* ── divider ── */
.tln-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONSTANTS & INFRASTRUCTURE CATALOG
# ══════════════════════════════════════════════════════════════════════════════

MAP_CENTER = (25.396, 68.357)   # Hyderabad, Sindh

INFRA_CATALOG = {
    "Macro Cell Tower": {
        "capex_pkr":    45_000_000,
        "max_subs":     10_000,
        "radius_km":    12,
        "opex_pct":     0.12,
        "icon":         "📡",
        "color":        "#00A8FF",
        "description":  "High range · High CAPEX · 45M PKR",
    },
    "Micro Cell Tower": {
        "capex_pkr":    15_000_000,
        "max_subs":     3_500,
        "radius_km":    5,
        "opex_pct":     0.12,
        "icon":         "🗼",
        "color":        "#FF9500",
        "description":  "Mid range · Mid CAPEX · 15M PKR",
    },
    "Small Cell / Urban Relay": {
        "capex_pkr":    5_000_000,
        "max_subs":     1_000,
        "radius_km":    1.5,
        "opex_pct":     0.12,
        "icon":         "📶",
        "color":        "#00D68F",
        "description":  "Short range · Low CAPEX · 5M PKR",
    },
}

TERRAIN_COLORS = {
    "Flat":           "#00D68F",
    "Urban":          "#00A8FF",
    "Hilly/Desolate": "#FF6B35",
}

DISCOUNT_RATE = 0.10   # for NPV / IRR calculation


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SYNTHETIC DATA ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_regional_data() -> pd.DataFrame:
    """
    Generate a mock geospatial DataFrame of 20 distinct telecom clusters
    centred around Hyderabad & rural Sindh, Pakistan.

    Returns
    -------
    pd.DataFrame with columns:
        cluster_id, location_name, lat, lon,
        population_density, deficiency_index,
        arpu_pkr, terrain, priority_score
    """
    rng = np.random.default_rng(42)

    location_names = [
        "Hyderabad Central",    "Latifabad",           "Qasimabad",
        "Kotri Industrial",      "Manjhand",            "Tando Allahyar",
        "Matiari",               "Hala",                "Sehwan Sharif",
        "Dadu District",         "Khairpur Nathan Shah","Mehar",
        "Johi",                  "Kamber",              "Shahdadkot",
        "Larkana North",         "Ratodero",            "Kandhkot Rural",
        "Jacobabad Periphery",   "Shikarpur Edge",
    ]

    # Cluster coordinates spread across Sindh from Hyderabad northward
    base_lat = np.array([
        25.396, 25.432, 25.371, 25.362, 25.530, 25.469, 25.597, 25.792,
        26.426, 26.731, 26.955, 27.161, 27.002, 27.588, 27.857, 27.556,
        27.800, 28.249, 28.688, 27.952,
    ])
    base_lon = np.array([
        68.357, 68.415, 68.330, 68.309, 68.109, 68.724, 68.454, 68.426,
        67.868, 67.776, 67.729, 67.929, 67.620, 67.743, 67.899, 68.212,
        68.287, 69.294, 68.440, 68.643,
    ])

    # Add minor jitter to avoid perfect grid
    jitter = rng.uniform(-0.018, 0.018, size=(20,))
    lat = np.clip(base_lat + jitter, 24.8, 29.5)
    lon = np.clip(base_lon + jitter, 67.0, 70.5)

    population_density = rng.integers(800, 55_000, size=20)
    deficiency_index   = np.round(rng.uniform(0.15, 1.0, size=20), 2)
    arpu_pkr           = rng.integers(250, 950, size=20)
    terrain_choices    = rng.choice(
        ["Flat", "Urban", "Hilly/Desolate"],
        size=20,
        p=[0.45, 0.35, 0.20],
    )

    # Priority score: higher deficiency + higher density → more urgent
    priority_score = np.round(
        deficiency_index * 0.6 + (population_density / population_density.max()) * 0.4, 3
    )

    df = pd.DataFrame({
        "cluster_id":        range(1, 21),
        "location_name":     location_names,
        "lat":               np.round(lat, 5),
        "lon":               np.round(lon, 5),
        "population_density":population_density,
        "deficiency_index":  deficiency_index,
        "arpu_pkr":          arpu_pkr,
        "terrain":           terrain_choices,
        "priority_score":    priority_score,
    })

    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  NETWORK OPTIMIZATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_optimization(
    df: pd.DataFrame,
    infra_type: str,
    budget_pkr: float,
    adoption_rate: float,
) -> pd.DataFrame:
    """
    Greedy CAPEX-constrained site-selection algorithm.

    Sorts clusters by descending priority_score (coverage urgency weighted by
    population), then greedily allocates towers until budget is exhausted.

    Parameters
    ----------
    df            : Regional cluster DataFrame.
    infra_type    : Selected infrastructure key from INFRA_CATALOG.
    budget_pkr    : Maximum CAPEX in PKR.
    adoption_rate : Fraction [0,1] of max subscribers expected to sign up.

    Returns
    -------
    pd.DataFrame of selected deployment sites with financial projections.
    """
    catalog      = INFRA_CATALOG[infra_type]
    unit_capex   = catalog["capex_pkr"]
    max_subs     = catalog["max_subs"]
    opex_pct     = catalog["opex_pct"]

    if unit_capex <= 0:
        st.error("Infrastructure CAPEX is zero – cannot proceed.")
        return pd.DataFrame()

    remaining_budget = budget_pkr
    results          = []

    # Sort by descending priority; take only deficit clusters worth deploying
    candidates = df[df["deficiency_index"] >= 0.30].sort_values(
        "priority_score", ascending=False
    )

    for _, row in candidates.iterrows():
        if remaining_budget < unit_capex:
            break

        # Effective subscribers = adoption rate × min(max_subs, pop_density proxy)
        # population_density used as a relative cap on addressable market
        addressable   = min(max_subs, int(row["population_density"] * 0.15))
        eff_subs      = max(1, int(addressable * adoption_rate))
        annual_rev    = eff_subs * row["arpu_pkr"] * 12
        annual_opex   = unit_capex * opex_pct
        net_annual    = annual_rev - annual_opex

        # Payback period (avoid div-by-zero)
        payback = (unit_capex / net_annual) if net_annual > 0 else float("inf")

        # Priority ranking label
        if row["priority_score"] >= 0.75:
            priority_label = "🔴 CRITICAL"
        elif row["priority_score"] >= 0.55:
            priority_label = "🟠 HIGH"
        elif row["priority_score"] >= 0.35:
            priority_label = "🔵 MEDIUM"
        else:
            priority_label = "🟢 LOW"

        results.append({
            "Rank":              len(results) + 1,
            "Location":          row["location_name"],
            "Latitude":          row["lat"],
            "Longitude":         row["lon"],
            "Terrain":           row["terrain"],
            "Deficiency Index":  row["deficiency_index"],
            "Target Population": row["population_density"],
            "Eff. Subscribers":  eff_subs,
            "ARPU (PKR)":        row["arpu_pkr"],
            "CAPEX (PKR M)":     unit_capex / 1_000_000,
            "Annual Rev (PKR M)":round(annual_rev / 1_000_000, 2),
            "Annual OPEX (PKR M)":round(annual_opex / 1_000_000, 2),
            "Net Annual (PKR M)":round(net_annual / 1_000_000, 2),
            "Payback (Years)":   round(payback, 1) if payback != float("inf") else "N/A",
            "Priority":          priority_label,
            "Priority Score":    row["priority_score"],
        })

        remaining_budget -= unit_capex

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FINANCIAL MODEL – NPV / IRR / CASH FLOW
# ══════════════════════════════════════════════════════════════════════════════

def compute_financial_model(
    deployment_df: pd.DataFrame,
    infra_type: str,
    adoption_rate: float,
    payback_years: int,
    discount_rate: float = DISCOUNT_RATE,
) -> dict:
    """
    Build a 5-year financial projection matrix from the deployment plan.

    Computes:
        - Annual Recurring Revenue (ARR)
        - OPEX (12 % of total CAPEX annualised)
        - EBITDA, Depreciation, NOPAT
        - Cumulative Cash Flow
        - Net Present Value (NPV)
        - Internal Rate of Return (IRR) via Newton–Raphson approximation

    Parameters
    ----------
    deployment_df : Output from run_optimization().
    infra_type    : Selected infrastructure type key.
    adoption_rate : Subscriber adoption fraction.
    payback_years : Target payback horizon (used in subscriber ramp model).
    discount_rate : WACC / hurdle rate for NPV.

    Returns
    -------
    dict with keys: years, arr, opex, ebitda, depreciation, nopat,
                    cum_cf, npv, irr, total_capex, total_subs_y,
                    congestion_y
    """
    if deployment_df.empty:
        return {}

    catalog    = INFRA_CATALOG[infra_type]
    unit_capex = catalog["capex_pkr"]
    useful_life = 10   # years – standard telecom tower useful life

    n_sites    = len(deployment_df)
    total_capex = n_sites * unit_capex
    annual_depr = total_capex / useful_life

    years = list(range(1, 6))

    # Subscriber ramp: logistic S-curve adoption over 5 years
    # Year 1 ramp starts low, saturates towards full adoption by ~year 5
    def sub_ramp(y: int) -> float:
        """Logistic growth factor for subscriber ramp."""
        k    = 1.8   # growth steepness
        mid  = 2.2   # inflection point (year)
        return 1 / (1 + math.exp(-k * (y - mid)))

    base_total_subs = deployment_df["Eff. Subscribers"].sum()
    base_avg_arpu   = deployment_df["ARPU (PKR)"].mean()
    annual_opex_total = total_capex * catalog["opex_pct"]

    arr         = []
    opex        = []
    ebitda      = []
    depreciation= []
    nopat       = []
    cash_flow   = []
    total_subs_y= []
    congestion_y= []

    for y in years:
        ramp         = sub_ramp(y)
        subs_y       = int(base_total_subs * ramp)
        rev_y        = subs_y * base_avg_arpu * 12
        opex_y       = annual_opex_total * (1 + 0.03 * (y - 1))   # 3 % OPEX inflation
        ebitda_y     = rev_y - opex_y
        nopat_y      = ebitda_y - annual_depr
        cf_y         = nopat_y + annual_depr   # add back non-cash depreciation

        # Congestion metric: network utilisation % (capped 100)
        max_cap = base_total_subs * 1.0   # full capacity at 100% adoption
        congestion = min(100, round((subs_y / max(1, max_cap)) * 100, 1))

        arr.append(round(rev_y / 1_000_000, 2))
        opex.append(round(opex_y / 1_000_000, 2))
        ebitda.append(round(ebitda_y / 1_000_000, 2))
        depreciation.append(round(annual_depr / 1_000_000, 2))
        nopat.append(round(nopat_y / 1_000_000, 2))
        cash_flow.append(round(cf_y / 1_000_000, 2))
        total_subs_y.append(subs_y)
        congestion_y.append(congestion)

    # Cumulative cash flow (Year 0 = -CAPEX)
    cum_cf = []
    running = -total_capex / 1_000_000
    for cf in cash_flow:
        running += cf
        cum_cf.append(round(running, 2))

    # NPV: sum of PV of each year's free cash flow minus initial CAPEX
    npv = -total_capex
    for i, cf in enumerate(cash_flow):
        npv += (cf * 1_000_000) / ((1 + discount_rate) ** (i + 1))
    npv = round(npv / 1_000_000, 2)

    # IRR via Newton–Raphson (simple bisection fallback)
    def _npv_at_rate(r: float) -> float:
        if r <= -1:
            return float("inf")
        v = -total_capex
        for i, cf in enumerate(cash_flow):
            v += (cf * 1_000_000) / ((1 + r) ** (i + 1))
        return v

    irr = None
    try:
        lo, hi = -0.5, 5.0
        for _ in range(200):
            mid = (lo + hi) / 2
            val = _npv_at_rate(mid)
            if abs(val) < 1:
                irr = round(mid * 100, 2)
                break
            if val > 0:
                lo = mid
            else:
                hi = mid
    except Exception:
        irr = None

    return {
        "years":        years,
        "arr":          arr,
        "opex":         opex,
        "ebitda":       ebitda,
        "depreciation": depreciation,
        "nopat":        nopat,
        "cash_flow":    cash_flow,
        "cum_cf":       cum_cf,
        "npv":          npv,
        "irr":          irr,
        "total_capex":  round(total_capex / 1_000_000, 2),
        "total_subs_y": total_subs_y,
        "congestion_y": congestion_y,
        "n_sites":      n_sites,
        "base_subs":    base_total_subs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_map(
    df: pd.DataFrame,
    deployment_df: pd.DataFrame | None = None,
    infra_type: str | None = None,
) -> folium.Map:
    """
    Construct a Folium interactive map with two optional layers:
        Layer 1 – Dead-zone markers (always shown)
        Layer 2 – Optimized deployment sites (shown post-optimization)

    Parameters
    ----------
    df            : Full regional cluster DataFrame.
    deployment_df : Optional deployment plan from run_optimization().
    infra_type    : Infrastructure type for popup icon selection.

    Returns
    -------
    folium.Map object ready for st_folium rendering.
    """
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=8,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # ── Layer 1: Dead-Zone Coverage Gap Markers ────────────────────────────
    dead_zone_group = folium.FeatureGroup(name="📍 Coverage Dead-Zones", show=True)

    for _, row in df.iterrows():
        di = row["deficiency_index"]

        # Color by deficiency severity
        if di >= 0.75:
            color, fill_color, severity_label = "#FF4D4D", "#FF4D4D", "CRITICAL"
        elif di >= 0.45:
            color, fill_color, severity_label = "#FF9500", "#FF9500", "MODERATE"
        else:
            color, fill_color, severity_label = "#00D68F", "#00D68F", "ACCEPTABLE"

        # Radius scaled by population density (normalised)
        pop_norm = (row["population_density"] / df["population_density"].max())
        radius   = 4000 + pop_norm * 14_000   # metres

        popup_html = f"""
        <div style="font-family:sans-serif;font-size:13px;min-width:200px">
          <b style="font-size:15px">{row['location_name']}</b><br>
          <hr style="margin:6px 0;border-color:#333">
          🏷️ <b>Severity:</b> {severity_label}<br>
          📊 <b>Deficiency Index:</b> {di:.2f}<br>
          👥 <b>Population Density:</b> {row['population_density']:,}<br>
          💰 <b>ARPU:</b> PKR {row['arpu_pkr']:,}<br>
          🗺️ <b>Terrain:</b> {row['terrain']}<br>
          ⚡ <b>Priority Score:</b> {row['priority_score']:.3f}
        </div>
        """

        folium.Circle(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.20,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{'🔴' if di >= 0.75 else '🟠' if di >= 0.45 else '🟢'} {row['location_name']} — Deficiency: {di:.2f}",
        ).add_to(dead_zone_group)

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.9,
            weight=1.5,
        ).add_to(dead_zone_group)

    dead_zone_group.add_to(m)

    # ── Layer 2: Optimized Deployment Sites ───────────────────────────────
    if deployment_df is not None and not deployment_df.empty:
        catalog      = INFRA_CATALOG.get(infra_type, list(INFRA_CATALOG.values())[0])
        deploy_group = folium.FeatureGroup(name="📡 Optimized Tower Placements", show=True)
        radius_m     = catalog["radius_km"] * 1000

        for _, site in deployment_df.iterrows():
            popup_html = f"""
            <div style="font-family:sans-serif;font-size:13px;min-width:220px">
              <b style="font-size:15px">{catalog['icon']} {site['Location']}</b><br>
              <hr style="margin:6px 0;border-color:#333">
              🏗️ <b>Type:</b> {infra_type}<br>
              💎 <b>Priority:</b> {site['Priority']}<br>
              💰 <b>CAPEX:</b> PKR {site['CAPEX (PKR M)']:.0f}M<br>
              👥 <b>Est. Subscribers:</b> {site['Eff. Subscribers']:,}<br>
              📈 <b>Annual Rev:</b> PKR {site['Annual Rev (PKR M)']}M<br>
              ⏱️ <b>Payback:</b> {site['Payback (Years)']} yrs
            </div>
            """

            # Pulsing coverage ring
            folium.Circle(
                location=[site["Latitude"], site["Longitude"]],
                radius=radius_m,
                color=catalog["color"],
                fill=True,
                fill_color=catalog["color"],
                fill_opacity=0.12,
                weight=2,
                dash_array="6 4",
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{catalog['icon']} {site['Location']} — Rank #{int(site['Rank'])}",
            ).add_to(deploy_group)

            # Tower marker
            folium.Marker(
                location=[site["Latitude"], site["Longitude"]],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{catalog['icon']} {site['Location']}",
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        background:{catalog['color']};
                        color:#000;
                        font-size:18px;
                        width:34px;height:34px;
                        border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        border:2px solid white;
                        box-shadow:0 0 10px {catalog['color']}88;
                        font-weight:bold;
                    ">{catalog['icon']}</div>""",
                    icon_size=(34, 34),
                    icon_anchor=(17, 17),
                ),
            ).add_to(deploy_group)

        deploy_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTLY CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#8BACC8", size=12),
    legend=dict(
        bgcolor="rgba(13,34,64,0.7)",
        bordercolor="rgba(0,168,255,0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(
        gridcolor="rgba(0,168,255,0.08)",
        zerolinecolor="rgba(0,168,255,0.15)",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(0,168,255,0.08)",
        zerolinecolor="rgba(0,168,255,0.15)",
        tickfont=dict(size=11),
    ),
)


def chart_cashflow(fin: dict) -> go.Figure:
    """
    Dual-axis chart: Cumulative Cash Flow (bar) + Break-Even Line (line).
    """
    years_lbl = [f"Year {y}" for y in fin["years"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Annual Net Cash Flow bars
    bar_colors = [
        "#00D68F" if v >= 0 else "#FF4D4D" for v in fin["cash_flow"]
    ]
    fig.add_trace(
        go.Bar(
            x=years_lbl,
            y=fin["cash_flow"],
            name="Annual Net Cash Flow (PKR M)",
            marker_color=bar_colors,
            marker_line_width=0,
            opacity=0.82,
        ),
        secondary_y=False,
    )

    # Cumulative Cash Flow line
    fig.add_trace(
        go.Scatter(
            x=years_lbl,
            y=fin["cum_cf"],
            mode="lines+markers",
            name="Cumulative Cash Flow (PKR M)",
            line=dict(color="#00A8FF", width=2.5),
            marker=dict(size=8, symbol="circle", color="#00A8FF",
                        line=dict(color="white", width=1.5)),
        ),
        secondary_y=True,
    )

    # Break-even zero line
    fig.add_hline(
        y=0, line_dash="dash", line_color="#FF6B35",
        line_width=1.5, secondary_y=True,
        annotation_text="Break-Even", annotation_font_color="#FF6B35",
        annotation_position="bottom right",
    )

    layout = {**PLOTLY_LAYOUT_BASE}
    layout["title"] = dict(
        text="Cumulative Cash Flow vs. Break-Even Horizon",
        font=dict(family="Syne, sans-serif", size=16, color="#E8F4FF"),
        x=0.02,
    )
    layout["yaxis"]  = {**layout["yaxis"],  "title": "Annual CF (PKR M)"}
    layout["yaxis2"] = dict(
        title="Cumulative CF (PKR M)",
        overlaying="y", side="right",
        gridcolor="rgba(0,168,255,0.05)",
        tickfont=dict(size=11, color="#8BACC8"),
        showgrid=False,
    )
    fig.update_layout(**layout)
    return fig


def chart_subscribers(fin: dict) -> go.Figure:
    """
    Grouped bar chart: Subscriber Acquisition vs Network Congestion by year.
    """
    years_lbl = [f"Year {y}" for y in fin["years"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=years_lbl,
            y=fin["total_subs_y"],
            name="Subscriber Acquisition",
            marker_color="#00A8FF",
            marker_line_width=0,
            opacity=0.85,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=years_lbl,
            y=fin["congestion_y"],
            mode="lines+markers",
            name="Network Utilisation (%)",
            line=dict(color="#FF9500", width=2.5, dash="dot"),
            marker=dict(size=9, symbol="diamond", color="#FF9500",
                        line=dict(color="white", width=1.5)),
        ),
        secondary_y=True,
    )

    layout = {**PLOTLY_LAYOUT_BASE}
    layout["title"] = dict(
        text="Subscriber Acquisition Trajectory & Network Utilisation",
        font=dict(family="Syne, sans-serif", size=16, color="#E8F4FF"),
        x=0.02,
    )
    layout["yaxis"]  = {**layout["yaxis"],  "title": "Subscribers"}
    layout["yaxis2"] = dict(
        title="Network Utilisation (%)",
        overlaying="y", side="right",
        range=[0, 110],
        gridcolor="rgba(0,168,255,0.05)",
        tickfont=dict(size=11, color="#8BACC8"),
        ticksuffix="%",
        showgrid=False,
    )
    fig.update_layout(**layout)
    return fig


def chart_revenue_breakdown(fin: dict) -> go.Figure:
    """
    Stacked area chart for ARR vs OPEX over 5 years.
    """
    years_lbl = [f"Year {y}" for y in fin["years"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years_lbl, y=fin["arr"],
        name="Annual Revenue (PKR M)",
        fill="tozeroy",
        line=dict(color="#00A8FF", width=2),
        fillcolor="rgba(0,168,255,0.18)",
    ))
    fig.add_trace(go.Scatter(
        x=years_lbl, y=fin["opex"],
        name="OPEX (PKR M)",
        fill="tozeroy",
        line=dict(color="#FF6B35", width=2),
        fillcolor="rgba(255,107,53,0.18)",
    ))
    fig.add_trace(go.Scatter(
        x=years_lbl, y=fin["ebitda"],
        name="EBITDA (PKR M)",
        line=dict(color="#00D68F", width=2.5, dash="dot"),
    ))

    layout = {**PLOTLY_LAYOUT_BASE}
    layout["title"] = dict(
        text="Revenue vs. OPEX vs. EBITDA (5-Year)",
        font=dict(family="Syne, sans-serif", size=16, color="#E8F4FF"),
        x=0.02,
    )
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8.  UI HELPER COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, sub: str = "", delta: str = "", delta_up: bool = True) -> str:
    """Return HTML for a single KPI glassmorphism card."""
    delta_class = "kpi-delta-up" if delta_up else "kpi-delta-down"
    delta_html  = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {delta_html}
    </div>
    """


def section_header(title: str, badge: str = "") -> None:
    badge_html = f'<span class="section-pill">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="section-header"><h2>{title}</h2>{badge_html}</div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown('<hr class="tln-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN APPLICATION LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Primary application entry-point."""

    # ── Load regional data ─────────────────────────────────────────────────
    df = generate_regional_data()

    # ══════════════════════════════════════════════════════════════════════
    # SIDEBAR – Infrastructure Configurator
    # ══════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("""
        <div class="brand-header">
            <div class="brand-dot">📡</div>
            <div>
                <div class="brand-title">Telenor Pakistan</div>
                <div class="brand-sub">CAPEX & Coverage Optimizer</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Infrastructure Configurator")
        st.markdown("---")

        infra_type = st.selectbox(
            "Tower Technology",
            options=list(INFRA_CATALOG.keys()),
            help="Select the infrastructure type to deploy.",
        )
        catalog = INFRA_CATALOG[infra_type]
        st.markdown(
            f"<small style='color:#8BACC8'>{catalog['icon']} {catalog['description']}</small>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        budget_m = st.slider(
            "Max CAPEX Budget (PKR M)",
            min_value=10,
            max_value=250,
            value=120,
            step=5,
            help="Total capital expenditure ceiling in PKR millions.",
        )
        budget_pkr = budget_m * 1_000_000

        adoption_pct = st.slider(
            "Projected Adoption Rate (%)",
            min_value=5,
            max_value=95,
            value=45,
            step=5,
            help="Expected % of addressable subscribers that will onboard.",
        )
        adoption_rate = adoption_pct / 100.0

        payback_years = st.slider(
            "Target Payback Period (Years)",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
        )

        st.markdown("---")
        max_sites = max(1, int(budget_pkr / catalog["capex_pkr"]))
        st.markdown(
            f"<div style='color:#8BACC8;font-size:13px'>"
            f"💡 Budget allows <b style='color:#00A8FF'>{max_sites}</b> {infra_type}(s)"
            f" @ PKR {catalog['capex_pkr'] / 1_000_000:.0f}M each."
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        run_button = st.button("🚀 Run Network Optimization Engine")

        st.markdown("---")
        st.markdown(
            "<small style='color:#4A6580'>📍 Focus Region: Hyderabad & Rural Sindh, Pakistan<br>"
            "Discount Rate: 10% WACC · 12% OPEX Standard</small>",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # TOP BRAND BAR
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:18px 0 24px 0;border-bottom:1px solid rgba(0,168,255,0.15);
                margin-bottom:28px">
        <div>
            <h1 style="margin:0;font-family:'Syne',sans-serif;font-size:26px;
                       color:#E8F4FF;letter-spacing:-0.5px">
                📡 Telenor Network CAPEX & Coverage Optimizer
            </h1>
            <p style="margin:4px 0 0;color:#8BACC8;font-size:13px">
                Hyderabad & Rural Sindh Regional Rollout Intelligence Platform
            </p>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
            <span style="background:rgba(0,214,143,0.15);color:#00D68F;
                         border:1px solid rgba(0,214,143,0.3);border-radius:20px;
                         padding:4px 14px;font-size:12px;font-weight:600">
                ● LIVE SIM
            </span>
            <span style="background:rgba(0,168,255,0.12);color:#00A8FF;
                         border:1px solid rgba(0,168,255,0.25);border-radius:20px;
                         padding:4px 14px;font-size:12px;font-weight:600">
                20 CLUSTERS
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SESSION STATE – Optimization Results
    # ══════════════════════════════════════════════════════════════════════
    if "deployment_df" not in st.session_state:
        st.session_state.deployment_df = None
        st.session_state.fin_model     = None
        st.session_state.infra_type    = None

    if run_button:
        with st.spinner("🔄 Running coverage optimization algorithm…"):
            st.session_state.deployment_df = run_optimization(
                df, infra_type, budget_pkr, adoption_rate
            )
            if not st.session_state.deployment_df.empty:
                st.session_state.fin_model = compute_financial_model(
                    st.session_state.deployment_df,
                    infra_type, adoption_rate, payback_years,
                )
                st.session_state.infra_type = infra_type

    dep  = st.session_state.deployment_df
    fin  = st.session_state.fin_model

    # ══════════════════════════════════════════════════════════════════════
    # KPI MATRIX CARDS
    # ══════════════════════════════════════════════════════════════════════
    section_header("Real-Time KPI Matrix", "LIVE")

    # Compute aggregate KPIs from regional data
    total_dead_zones = len(df[df["deficiency_index"] >= 0.75])
    avg_deficiency   = df["deficiency_index"].mean()
    avg_arpu         = df["arpu_pkr"].mean()
    total_pop        = df["population_density"].sum()

    kpi_cols = st.columns(5)
    kpi_data = [
        ("MONITORED CLUSTERS",   "20",                 "Hyderabad & Sindh", "↗ Full Coverage", True),
        ("CRITICAL DEAD ZONES",  str(total_dead_zones),"Deficiency ≥ 0.75",  f"▲ {total_dead_zones} urgent", False),
        ("AVG DEFICIENCY INDEX", f"{avg_deficiency:.2f}","Regional signal gap", "▼ Requires action", False),
        ("AVG ARPU",             f"PKR {avg_arpu:,.0f}","Monthly revenue/user","↗ Market potential", True),
        ("TOTAL ADDRESSABLE POP",f"{total_pop/1000:.0f}K","Aggregate density",  "↗ Growth ready", True),
    ]
    for col, (label, value, sub, delta, up) in zip(kpi_cols, kpi_data):
        col.markdown(kpi_card(label, value, sub, delta, up), unsafe_allow_html=True)

    if dep is not None and not dep.empty and fin:
        divider()
        opt_kpi_cols = st.columns(5)
        irr_str = f"{fin['irr']:.1f}%" if fin["irr"] is not None else "N/A"
        npv_sign = "↗ Positive ROI" if fin["npv"] >= 0 else "↘ Below Hurdle"
        opt_kpi = [
            ("SITES DEPLOYED",       str(fin["n_sites"]),          infra_type,                   "↗ Deployed", True),
            ("TOTAL CAPEX",          f"PKR {fin['total_capex']:.0f}M","Capital deployed",         "Budget locked", True),
            ("5-YR NPV",             f"PKR {fin['npv']:.1f}M",      "@ 10% discount rate",        npv_sign, fin["npv"] >= 0),
            ("PROJ. IRR",            irr_str,                       "Internal rate of return",    "↗ vs WACC 10%", fin["irr"] is not None and fin["irr"] > 10),
            ("YEAR-5 SUBSCRIBERS",   f"{fin['total_subs_y'][-1]:,}","Estimated active base",      "↗ S-curve ramp", True),
        ]
        for col, (label, value, sub, delta, up) in zip(opt_kpi_cols, opt_kpi):
            col.markdown(kpi_card(label, value, sub, delta, up), unsafe_allow_html=True)

    divider()

    # ══════════════════════════════════════════════════════════════════════
    # GEOSPATIAL MAP
    # ══════════════════════════════════════════════════════════════════════
    section_header("Geospatial Coverage Intelligence Map", "INTERACTIVE")

    map_col, legend_col = st.columns([3, 1])

    with map_col:
        fmap = build_map(df, dep, st.session_state.infra_type)
        st_folium(fmap, width=None, height=540, returned_objects=[])

    with legend_col:
        st.markdown("""
        <div class="glass-panel" style="height:100%">
            <div style="font-family:'Syne',sans-serif;font-size:14px;
                        font-weight:700;color:#E8F4FF;margin-bottom:16px">
                🗺️ Map Legend
            </div>
            <div style="font-size:12px;color:#8BACC8;line-height:2.0">
                <div>🔴 <b style="color:#FF4D4D">Critical Deficiency</b><br>
                    <small>Index ≥ 0.75 · Immediate action</small></div>
                <br>
                <div>🟠 <b style="color:#FF9500">Moderate Deficiency</b><br>
                    <small>Index 0.45–0.75 · Planned upgrade</small></div>
                <br>
                <div>🟢 <b style="color:#00D68F">Acceptable Coverage</b><br>
                    <small>Index &lt; 0.45 · Monitoring</small></div>
                <br>
                <hr style="border-color:rgba(0,168,255,0.15);margin:10px 0">
                <div><b style="color:#00A8FF">Circle Radius</b><br>
                    <small>Proportional to population density</small></div>
                <br>
                <div><b style="color:#E8F4FF">Dashed Ring</b><br>
                    <small>Tower coverage footprint</small></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if dep is None:
        st.info(
            "ℹ️ Configure your infrastructure parameters in the sidebar and click "
            "**🚀 Run Network Optimization Engine** to overlay optimal tower placements.",
            icon="📡",
        )

    divider()

    # ══════════════════════════════════════════════════════════════════════
    # FINANCIAL ANALYTICS PANEL
    # ══════════════════════════════════════════════════════════════════════
    if dep is not None and not dep.empty and fin:
        section_header("Predictive ROI & Financial Deep-Dive", "5-YEAR MODEL")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(chart_cashflow(fin), use_container_width=True)
        with chart_col2:
            st.plotly_chart(chart_subscribers(fin), use_container_width=True)

        st.plotly_chart(chart_revenue_breakdown(fin), use_container_width=True)

        # 5-Year Financial Summary Table
       # 5-Year Financial Summary Table
        with st.expander("📊 5-Year Financial Matrix Detail", expanded=False):
            fin_table = pd.DataFrame({
                "Year":              [f"Year {y}" for y in fin["years"]],
                "ARR (PKR M)":       fin["arr"],
                "OPEX (PKR M)":      fin["opex"],
                "EBITDA (PKR M)":    fin["ebitda"],
                "Depreciation (M)":  fin["depreciation"],
                "NOPAT (PKR M)":     fin["nopat"],
                "Free CF (PKR M)":   fin["cash_flow"],
                "Cumulative CF (M)": fin["cum_cf"],
                "Subscribers":       fin["total_subs_y"],
                "Network Util (%)":  fin["congestion_y"],
            })
            st.dataframe(
                fin_table.style
                .format(precision=2)
                .map(
                    lambda v: "color: #00D68F" if isinstance(v, (int, float)) and v >= 0
                              else "color: #FF4D4D",
                    subset=["EBITDA (PKR M)", "NOPAT (PKR M)", "Free CF (PKR M)", "Cumulative CF (M)"],
                ),
                use_container_width=True,
                hide_index=True,
            )

        divider()

        # ══════════════════════════════════════════════════════════════════
        # FIELD DEPLOYMENT MANIFEST
        # ══════════════════════════════════════════════════════════════════
        section_header("Field Deployment Manifest", "ENGINEERING OUTPUT")

        display_cols = [
            "Rank", "Priority", "Location", "Terrain",
            "Deficiency Index", "Target Population", "Eff. Subscribers",
            "ARPU (PKR)", "CAPEX (PKR M)", "Annual Rev (PKR M)",
            "Net Annual (PKR M)", "Payback (Years)",
            "Latitude", "Longitude",
        ]

        manifest = dep[display_cols].copy()

        def highlight_priority(row):
            styles = [""] * len(row)
            p = str(row.get("Priority", ""))
            if "CRITICAL" in p:
                styles[1] = "color: #FF4D4D; font-weight: bold"
            elif "HIGH" in p:
                styles[1] = "color: #FF9500; font-weight: bold"
            elif "MEDIUM" in p:
                styles[1] = "color: #00A8FF; font-weight: bold"
            else:
                styles[1] = "color: #00D68F; font-weight: bold"
            return styles

        st.dataframe(
            manifest.style.apply(highlight_priority, axis=1).format({
                "Deficiency Index": "{:.2f}",
                "CAPEX (PKR M)":    "{:.0f}",
                "Annual Rev (PKR M)":"{:.2f}",
                "Net Annual (PKR M)":"{:.2f}",
                "Latitude":         "{:.5f}",
                "Longitude":        "{:.5f}",
            }),
            use_container_width=True,
            hide_index=True,
            height=340,
        )

        # ── CSV Download ──────────────────────────────────────────────────
        csv_buf = io.StringIO()
        dep.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        dl_col, stat_col = st.columns([1, 2])
        with dl_col:
            st.download_button(
                label="⬇️ Download Full Manifest (CSV)",
                data=csv_bytes,
                file_name=f"telenor_deployment_manifest_{infra_type.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with stat_col:
            st.markdown(
                f"""
                <div class="glass-panel" style="padding:14px 18px">
                    <div style="display:flex;gap:32px;flex-wrap:wrap">
                        <div>
                            <div class="kpi-label">SITES IN MANIFEST</div>
                            <div style="font-family:'Syne',sans-serif;font-size:22px;
                                        font-weight:800;color:#00A8FF">{len(dep)}</div>
                        </div>
                        <div>
                            <div class="kpi-label">TOTAL CAPEX DEPLOYED</div>
                            <div style="font-family:'Syne',sans-serif;font-size:22px;
                                        font-weight:800;color:#00A8FF">
                                PKR {dep['CAPEX (PKR M)'].sum():.0f}M
                            </div>
                        </div>
                        <div>
                            <div class="kpi-label">TOTAL SUBSCRIBERS</div>
                            <div style="font-family:'Syne',sans-serif;font-size:22px;
                                        font-weight:800;color:#00D68F">
                                {dep['Eff. Subscribers'].sum():,}
                            </div>
                        </div>
                        <div>
                            <div class="kpi-label">AVG PAYBACK</div>
                            <div style="font-family:'Syne',sans-serif;font-size:22px;
                                        font-weight:800;color:#FF9500">
                                {dep[dep['Payback (Years)'] != 'N/A']['Payback (Years)'].astype(float).mean():.1f} yrs
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif dep is not None and dep.empty:
        st.warning(
            "⚠️ The optimization engine found no viable sites within the current budget and parameters. "
            "Try increasing the CAPEX budget or selecting a lower-cost infrastructure type.",
            icon="💡",
        )

    # ══════════════════════════════════════════════════════════════════════
    # REGIONAL DATA EXPLORER
    # ══════════════════════════════════════════════════════════════════════
    with st.expander("🔍 Regional Cluster Data Explorer", expanded=False):
        section_header("All 20 Regional Clusters", "RAW DATA")
        st.dataframe(
            df.rename(columns={
                "cluster_id":         "Cluster ID",
                "location_name":      "Location",
                "lat":                "Latitude",
                "lon":                "Longitude",
                "population_density": "Population Density",
                "deficiency_index":   "Deficiency Index",
                "arpu_pkr":           "ARPU (PKR)",
                "terrain":            "Terrain",
                "priority_score":     "Priority Score",
            }).style.background_gradient(
                subset=["Deficiency Index", "Priority Score"],
                cmap="RdYlGn_r",
            ).format({
                "Deficiency Index": "{:.2f}",
                "Priority Score":   "{:.3f}",
                "Latitude":         "{:.5f}",
                "Longitude":        "{:.5f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # ── Footer ─────────────────────────────────────────────────────────────
    divider()
    st.markdown("""
    <div style="text-align:center;padding:16px 0;color:#3A5A7C;font-size:11px;
                letter-spacing:1px;text-transform:uppercase">
        Telenor Pakistan · CAPEX & Coverage Intelligence Platform ·
        Hyderabad & Rural Sindh Network Planning Division ·
        All data synthetically generated for simulation purposes
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
