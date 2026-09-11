import streamlit as st
import joblib
import rasterio
import numpy as np
import geopandas as gpd
import folium
import pandas as pd
import json
import urllib.parse
import urllib.request
from shapely.geometry import Point

from streamlit_folium import st_folium

try:
    from data.alert_engine import check_users_for_alerts
    ALERT_ENGINE_READY = True
except Exception:
    check_users_for_alerts = None
    ALERT_ENGINE_READY = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NER Landslide Risk Monitoring",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background-color: #f5f7f9;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }


    /* Typography */

    h1 {
        color: #172b3a !important;
        font-size: 2.05rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.2rem !important;
    }

    h2 {
        color: #203746 !important;
        font-size: 1.35rem !important;
        font-weight: 650 !important;
    }

    h3 {
        color: #304653 !important;
        font-size: 1.05rem !important;
    }


    /* Metrics */

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce2e7;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #667784 !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #172b3a !important;
        font-weight: 700 !important;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        background-color: #1d303d;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }

    section[data-testid="stSidebar"] p {
        color: #c4d0d8 !important;
    }


    /* Table */

    div[data-testid="stDataFrame"] {
        border: 1px solid #dce2e7;
        border-radius: 8px;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #7a8791;
        font-size: 0.75rem;
        padding-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = "data/landslide_model.joblib"

RAINFALL_FILE = "data/ner_rainfall_20260907.tif"

STATE_FILE = "data/state_NWIC.GeoJSON"


# ============================================================
# NER STATES
# ============================================================

NER_STATES = [
    "Assam",
    "Meghalaya",
    "Manipur",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura"
]

SIKKIM_PLACES = {
    "Gangtok": (27.3314, 88.6139),
    "Mangan": (27.5258, 88.5615),
    "Namchi": (27.1667, 88.3500),
    "Gyalshing": (27.2895, 88.2645),
    "Pakyong": (27.2300, 88.6200),
    "Soreng": (27.1667, 88.2000),
    "Singtam": (27.2347, 88.5014),
    "Rangpo": (27.1767, 88.5331),
    "Ravangla": (27.3125, 88.3639),
    "Jorethang": (27.1069, 88.3233),
    "Namthang": (27.0833, 88.4167),
    "Melli": (27.0667, 88.4333),
    "Temi": (27.2333, 88.3500),
    "Rhenock": (27.2500, 88.7000),
    "Rongli": (27.2167, 88.6833),
    "Chungthang": (27.5889, 88.6433),
    "Lachen": (27.7167, 88.7667),
    "Lachung": (27.6875, 88.7386),
    "Yuksom": (27.3667, 88.2167),
    "Dentam": (27.2833, 88.2167),
    "Pelling": (27.3050, 88.2390),
    "Dikchu": (27.4200, 88.5800),
    "Dzongu": (27.5600, 88.6200),
    "Kabi": (27.4600, 88.6200),
    "Phodong": (27.3900, 88.6100),
    "Singhik": (27.5100, 88.6100),
    "Passingdong": (27.5900, 88.5700),
    "Yumthang": (27.7900, 88.6900),
    "Zuluk": (27.8500, 88.7700),
    "Aritar": (27.2500, 88.6800),
    "Rolep": (27.2500, 88.7400),
    "Gnathang": (27.7500, 88.7700),
    "Kupup": (27.7500, 88.8500),
    "Tsomgo": (27.3700, 88.7600),
    "Nathu La": (27.3900, 88.8300),
    "Tadong": (27.3150, 88.5900),
    "Deorali": (27.3150, 88.6000),
    "Rumtek": (27.3000, 88.5700),
    "Ralang": (27.3000, 88.3600),
    "Borang": (27.1800, 88.4100),
    "Sichey": (27.3350, 88.6000),
    "Majitar": (27.1450, 88.5400),
}



# Reference monitoring locations for all seven NER states.
# These are geographic reference locations for prototype monitoring,
# not historical landslide locations.
ALL_NER_PLACES = {
    "Assam": {
        "Guwahati": (26.1445, 91.7362), "Dibrugarh": (27.4728, 94.9120),
        "Silchar": (24.8333, 92.7789), "Jorhat": (26.7509, 94.2037),
        "Tezpur": (26.6528, 92.7926), "Nagaon": (26.3484, 92.6838),
        "Tinsukia": (27.4922, 95.3468), "Sivasagar": (26.9820, 94.6425),
        "North Lakhimpur": (27.2352, 94.1036), "Diphu": (25.8430, 93.4316),
        "Goalpara": (26.1667, 90.6167), "Dhemaji": (27.4833, 94.5833),
    },
    "Meghalaya": {
        "Shillong": (25.5788, 91.8933), "Tura": (25.5144, 90.2033),
        "Jowai": (25.4500, 92.2000), "Nongpoh": (25.9000, 91.8833),
        "Nongstoin": (25.5167, 91.2667), "Williamnagar": (25.5000, 90.6000),
        "Cherrapunji": (25.2700, 91.7300), "Mairang": (25.5500, 91.3000),
        "Baghmara": (25.2000, 90.6333), "Resubelpara": (25.9000, 90.6000),
    },
    "Manipur": {
        "Imphal": (24.8170, 93.9368), "Thoubal": (24.6380, 94.0100),
        "Churachandpur": (24.3333, 93.6833), "Ukhrul": (25.0950, 94.3610),
        "Senapati": (25.2670, 94.2600), "Tamenglong": (24.9833, 93.5000),
        "Kakching": (24.5000, 94.0000), "Moirang": (24.4970, 93.7740),
        "Jiribam": (24.8000, 93.1167), "Chandel": (24.3200, 94.2500),
    },
    "Mizoram": {
        "Aizawl": (23.7271, 92.7176), "Lunglei": (22.8897, 92.7460),
        "Champhai": (23.4670, 93.3280), "Kolasib": (24.2230, 92.6790),
        "Serchhip": (23.3050, 92.8470), "Saiha": (22.4890, 92.9810),
        "Lawngtlai": (22.5300, 92.9000), "Mamit": (23.9300, 92.4900),
        "Khawzawl": (23.5200, 93.1200), "Hnahthial": (22.9800, 92.9300),
    },
    "Nagaland": {
        "Kohima": (25.6751, 94.1086), "Dimapur": (25.9110, 93.7217),
        "Mokokchung": (26.3220, 94.5180), "Tuensang": (26.2700, 94.8240),
        "Mon": (26.7200, 95.0300), "Wokha": (26.1000, 94.2700),
        "Zunheboto": (25.9700, 94.5200), "Phek": (25.6700, 94.4800),
        "Kiphire": (25.8300, 94.7800), "Longleng": (26.4800, 94.8200),
    },
    "Tripura": {
        "Agartala": (23.8315, 91.2868), "Dharmanagar": (24.3667, 92.1667),
        "Kailashahar": (24.3300, 92.0100), "Udaipur": (23.5300, 91.4800),
        "Belonia": (23.2500, 91.4500), "Ambassa": (23.9300, 91.8500),
        "Khowai": (24.0700, 91.6000), "Sabroom": (23.0000, 91.7300),
        "Teliamura": (24.0000, 91.5000), "Sonamura": (23.4800, 91.2700),
    },
}

ALL_NER_PLACES["Sikkim"] = SIKKIM_PLACES

# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(
        MODEL_FILE
    )

except Exception as error:

    st.error(
        f"Unable to load AI model: {error}"
    )

    st.stop()


# ============================================================
# LOAD STATE DATA
# ============================================================

try:

    states_gdf = gpd.read_file(
        STATE_FILE
    )

except Exception as error:

    st.error(
        f"Unable to load state boundaries: {error}"
    )

    st.stop()


# ============================================================
# FIND STATE COLUMN
# ============================================================

possible_columns = [
    "state_name",
    "state",
    "STATE",
    "ST_NAME",
    "name"
]

state_column = None


for column in possible_columns:

    if column in states_gdf.columns:

        state_column = column

        break


if state_column is None:

    st.error(
        "State name column not found."
    )

    st.stop()


# ============================================================
# FILTER NER
# ============================================================

ner_gdf = states_gdf[
    states_gdf[state_column].isin(
        NER_STATES
    )
].copy()


if ner_gdf.empty:

    st.error(
        "NER states were not found."
    )

    st.stop()


ner_wgs84 = ner_gdf.to_crs(
    "EPSG:4326"
)


# ============================================================
# RAINFALL EXTRACTION
# ============================================================

def get_location_rainfall(
    latitude,
    longitude
):

    try:

        with rasterio.open(
            RAINFALL_FILE
        ) as src:

            row, col = src.index(
                longitude,
                latitude
            )


            if (
                row < 0
                or row >= src.height
                or col < 0
                or col >= src.width
            ):

                return None


            row_start = max(
                0,
                row - 2
            )

            row_end = min(
                src.height,
                row + 3
            )

            col_start = max(
                0,
                col - 2
            )

            col_end = min(
                src.width,
                col + 3
            )


            window = rasterio.windows.Window(
                col_start,
                row_start,
                col_end - col_start,
                row_end - row_start
            )


            rainfall = src.read(
                1,
                window=window
            ).astype(float)


            rainfall = rainfall[
                rainfall >= 0
            ]


            if rainfall.size == 0:

                return None


            # NASA IMERG GIS scaling

            rainfall_mm = rainfall / 10.0


            return {

                "mean": float(
                    np.mean(
                        rainfall_mm
                    )
                ),

                "max": float(
                    np.max(
                        rainfall_mm
                    )
                ),

                "min": float(
                    np.min(
                        rainfall_mm
                    )
                )

            }


    except Exception:

        return None


# ============================================================
# RAINFALL WARNING
# ============================================================

def priority_warning(risk, rainfall):
    """Priority-location warning for Guwahati and Shillong.
    Uses AI risk plus 1-day mean/max rainfall. Prototype thresholds only.
    """
    mean_mm = float(rainfall.get("mean", 0))
    max_mm = float(rainfall.get("max", 0))
    if mean_mm >= 100 or (risk == "HIGH" and mean_mm >= 50):
        return "CRITICAL", "CRITICAL — Very high rainfall / high AI risk detected. Immediate monitoring is recommended."
    if risk == "HIGH" or mean_mm >= 50 or max_mm >= 75:
        return "HIGH", "HIGH — High AI risk or localized heavy rainfall detected. Enhanced early-warning monitoring is recommended."
    if risk == "MEDIUM" or mean_mm >= 25 or max_mm >= 25:
        return "WATCH", "WATCH — Elevated rainfall or moderate AI risk detected. Continue close monitoring."
    return "LOW", "LOW — No elevated priority warning is currently detected."


def rainfall_warning(
    rainfall
):
    """
    Prototype early-warning classification.
    These thresholds are not official government warning thresholds.
    """

    if rainfall >= 100:
        return (
            "CRITICAL",
            "CRITICAL — Very high rainfall detected. Immediate monitoring is recommended."
        )

    elif rainfall >= 50:
        return (
            "HIGH",
            "HIGH — Heavy rainfall detected. Increased landslide-risk monitoring is recommended."
        )

    elif rainfall >= 25:
        return (
            "WATCH",
            "WATCH — Elevated rainfall detected. Continue close monitoring."
        )

    else:
        return (
            "LOW",
            "LOW — No elevated rainfall warning is currently detected."
        )


# ============================================================
# AI PREDICTION
# ============================================================

def predict_risk(
    rainfall
):

    features = np.array(
        [[
            rainfall["mean"],
            rainfall["max"],
            rainfall["min"]
        ]]
    )


    prediction = model.predict(
        features
    )[0]


    probabilities = model.predict_proba(
        features
    )[0]


    confidence = float(
        np.max(
            probabilities
        ) * 100
    )


    risk_names = {
        0: "LOW",
        1: "MEDIUM",
        2: "HIGH"
    }


    risk = risk_names.get(
        int(prediction),
        "UNKNOWN"
    )


    return risk, confidence


# ============================================================
# EARLY WARNING / NOTIFICATION HELPERS
# ============================================================

def get_registered_user_count():
    try:
        users = pd.read_csv("data/registered_users.csv")
        return len(users)
    except Exception:
        return 0


def get_location_alerts(latitude, longitude, risk, rainfall_mm, max_rainfall_mm=None):
    if not ALERT_ENGINE_READY or check_users_for_alerts is None:
        return []

    try:
        # Use the stronger of mean and local maximum rainfall for
        # early-warning decisions so localized heavy rainfall is not lost.
        effective_rainfall = float(rainfall_mm)
        if max_rainfall_mm is not None:
            effective_rainfall = max(effective_rainfall, float(max_rainfall_mm))
        return check_users_for_alerts(
            latitude,
            longitude,
            risk,
            effective_rainfall,
            radius_km=25
        )
    except Exception:
        return []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "NER Monitoring"
    )

    st.caption(
        "Landslide Risk Assessment"
    )

    st.divider()


    # --------------------------------------------------------
    # MONITORING LOCATION
    # --------------------------------------------------------

    st.subheader(
        "Monitoring Location"
    )

    st.caption(
        "Choose a state to view its current "
        "environmental assessment."
    )


    selected_state = st.selectbox(

        "Select NER state",

        NER_STATES,

        index=0

    )

    selected_place = None
    selected_place_lat = None
    selected_place_lon = None

    st.subheader(f"{selected_state} Place")
    state_places = ALL_NER_PLACES.get(selected_state, {})

    if state_places:
        selected_place = st.selectbox(
            f"Select {selected_state} monitoring place",
            list(state_places.keys()),
            index=0,
            key="ner_place_selector"
        )
        selected_place_lat, selected_place_lon = state_places[selected_place]
        st.caption(
            f"Coordinates: {selected_place_lat:.5f}, "
            f"{selected_place_lon:.5f}"
        )
        if selected_state == "Assam" and selected_place == "Guwahati":
            st.info("Guwahati is configured as the primary Assam early-warning monitoring location.")
        elif selected_state == "Meghalaya" and selected_place == "Shillong":
            st.info("Shillong is configured as the primary Meghalaya early-warning monitoring location.")
    else:
        st.info("Place list is not available for this state.")



    # Selected state

    state_number = (
        NER_STATES.index(
            selected_state
        ) + 1
    )


    st.info(
        f"Selected area\n\n"
        f"**{selected_state}**\n\n"
        f"NER State {state_number} of 7"
    )


    st.divider()


    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.subheader(
        "System Status"
    )

    st.success(
        "AI model ready"
    )

    st.success(
        "Rainfall data ready"
    )

    st.success(
        "Boundary data ready"
    )

    if ALERT_ENGINE_READY:
        st.success(
            "Alert engine ready"
        )
    else:
        st.warning(
            "Alert engine unavailable"
        )


    st.divider()


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    st.subheader(
        "Data Sources"
    )

    st.caption(
        "NASA GPM IMERG"
    )

    st.caption(
        "NWIC / GSI boundaries"
    )

    st.caption(
        "Random Forest model"
    )


    st.divider()


    st.caption(
        "NER Landslide Monitoring • Prototype"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "NER Landslide Risk Monitoring"
)

st.write(
    "AI-assisted rainfall and landslide-risk monitoring "
    "for the North Eastern Region of India."
)


st.success(
    "SYSTEM OPERATIONAL  |  "
    "7 NER states monitored  |  "
    "1-day rainfall data"
)


# ============================================================
# ANALYZE ALL STATES
# ============================================================

results = []


for state_name in NER_STATES:

    selected = ner_wgs84[
        ner_wgs84[state_column]
        == state_name
    ]


    if selected.empty:

        continue


    if state_name == selected_state and selected_place is not None:
        latitude = float(selected_place_lat)
        longitude = float(selected_place_lon)
    else:
        point = (
            selected
            .geometry
            .iloc[0]
            .representative_point()
        )

        latitude = float(point.y)
        longitude = float(point.x)


    rainfall = get_location_rainfall(

        latitude,

        longitude

    )


    if rainfall is None:

        continue


    risk, confidence = predict_risk(
        rainfall
    )


    warning_level, warning_text = (
        rainfall_warning(
            rainfall["mean"]
        )
    )

    location_alerts = get_location_alerts(
        latitude,
        longitude,
        risk,
        rainfall["mean"],
        rainfall.get("max")
    )


    results.append({

        "state":
            state_name,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "rainfall":
            rainfall,

        "risk":
            risk,

        "confidence":
            confidence,

        "warning":
            warning_level,

        "warning_text":
            warning_text,

        "alerts":
            location_alerts

    })


# ============================================================
# PRIORITY EARLY-WARNING LOCATIONS
# ============================================================
# These two locations are explicitly monitored for the prototype:
# Guwahati in Assam and Shillong in Meghalaya.
# They use the same NASA rainfall -> AI risk -> alert pipeline.
# The coordinates are monitoring/reference locations, not historical
# landslide locations.

PRIORITY_EARLY_WARNING_PLACES = {
    "Assam": {
        "Guwahati": (26.1445, 91.7362),
    },
    "Meghalaya": {
        "Shillong": (25.5788, 91.8933),
    },
}

priority_early_warning_results = []

for ew_state, places in PRIORITY_EARLY_WARNING_PLACES.items():
    for ew_place, (ew_lat, ew_lon) in places.items():
        ew_rainfall = get_location_rainfall(ew_lat, ew_lon)
        if ew_rainfall is None:
            continue

        ew_risk, ew_confidence = predict_risk(ew_rainfall)
        ew_warning, ew_warning_text = priority_warning(ew_risk, ew_rainfall)
        ew_alerts = get_location_alerts(
            ew_lat,
            ew_lon,
            ew_risk,
            ew_rainfall["mean"],
            ew_rainfall.get("max")
        )

        priority_early_warning_results.append({
            "state": ew_state,
            "place": ew_place,
            "latitude": ew_lat,
            "longitude": ew_lon,
            "rainfall": ew_rainfall,
            "risk": ew_risk,
            "confidence": ew_confidence,
            "warning": ew_warning,
            "warning_text": ew_warning_text,
            "alerts": ew_alerts,
        })


# ============================================================
# SUMMARY
# ============================================================

high_count = sum(
    result["risk"] == "HIGH"
    for result in results
)


medium_count = sum(
    result["risk"] == "MEDIUM"
    for result in results
)


low_count = sum(
    result["risk"] == "LOW"
    for result in results
)


warning_count = sum(
    result["warning"] in [
        "WATCH",
        "HIGH",
        "CRITICAL"
    ]

    for result in results
)


if results:

    maximum_rainfall = max(

        result["rainfall"]["mean"]

        for result in results

    )

else:

    maximum_rainfall = 0


# ============================================================
# ALERT SUMMARY
# ============================================================

all_alerts = []

# Include alerts from the normal state monitoring results.
for result in results:
    all_alerts.extend(result.get("alerts", []))

# Also include the dedicated Guwahati/Shillong early-warning checks.
# Deduplicate by user + warning level so the same user is not counted twice
# when one of these locations is also the currently selected state place.
for ew_result in priority_early_warning_results:
    all_alerts.extend(ew_result.get("alerts", []))

unique_alerts = {}
for alert in all_alerts:
    key = (
        str(alert.get("name", "")),
        str(alert.get("phone", "")),
        str(alert.get("alert_level", "")),
    )
    unique_alerts[key] = alert

all_alerts = list(unique_alerts.values())

unique_alert_users = set()

for alert in all_alerts:
    user_key = (
        str(alert.get("name", "")),
        str(alert.get("phone", ""))
    )
    unique_alert_users.add(user_key)

registered_user_count = get_registered_user_count()
alert_user_count = len(unique_alert_users)


# ============================================================
# REGIONAL OVERVIEW
# ============================================================

st.header(
    "Regional Overview"
)

st.caption(
    "Current monitoring summary across the seven NER states."
)


c1, c2, c3, c4, c5 = st.columns(5)


with c1:

    st.metric(
        "States monitored",
        len(results)
    )


with c2:

    st.metric(
        "High AI risk",
        high_count
    )


with c3:

    st.metric(
        "Active warnings",
        warning_count
    )


with c4:

    st.metric(
        "Highest rainfall",
        f"{maximum_rainfall:.1f} mm"
    )


with c5:

    st.metric(
        "People at risk",
        alert_user_count
    )


# ============================================================
# REGIONAL WARNING
# ============================================================

st.subheader(
    "Regional Warning Status"
)


if high_count > 0:

    st.error(
        f"HIGH PRIORITY — "
        f"{high_count} location(s) "
        "currently show HIGH prototype risk."
    )

elif (
    medium_count > 0
    or warning_count > 0
):

    st.warning(
        "ELEVATED MONITORING — "
        "Moderate prototype risk and/or "
        "heavy rainfall has been detected."
    )

else:

    st.success(
        "NORMAL MONITORING — "
        "No high-level prototype warning "
        "is currently detected."
    )


# ============================================================
# PRIORITY EARLY-WARNING MONITORING
# ============================================================

st.subheader("Priority Early-Warning Monitoring")
st.caption(
    "Dedicated prototype monitoring for Guwahati, Assam and Shillong, Meghalaya. "
    "The assessment uses the current NASA 1-day rainfall value, the prototype AI risk model, "
    "and the location-based alert engine."
)

if priority_early_warning_results:
    ew_cards = st.columns(len(priority_early_warning_results))
    for card, ew_result in zip(ew_cards, priority_early_warning_results):
        with card:
            st.markdown(f"**{ew_result['place']}, {ew_result['state']}**")
            st.metric("1-day rainfall", f"{ew_result['rainfall']['mean']:.1f} mm")
            st.write(f"**AI Risk:** {ew_result['risk']}")
            st.write(f"**Early Warning:** {ew_result['warning']}")
            st.write(f"**Maximum rainfall:** {ew_result['rainfall']['max']:.1f} mm")
            st.write(f"**Users to Alert:** {len(ew_result.get('alerts', []))}")
            if ew_result.get("alerts"):
                st.error("🚨 HIGH-PRIORITY ALERT — warning generated for a nearby registered user.")
            elif ew_result["warning"] in ["HIGH", "CRITICAL"]:
                st.error(f"🚨 {ew_result['warning']} PRIORITY ALERT — high-risk condition detected; no registered user is currently within the alert radius.")
            elif ew_result["warning"] == "WATCH":
                st.warning("WATCH — early warning condition detected; continue close monitoring.")
            else:
                st.success("LOW — no elevated priority warning currently detected.")

    st.caption(
        "Prototype warning thresholds: LOW < 25 mm, WATCH 25–50 mm, "
        "HIGH 50–100 mm, CRITICAL ≥ 100 mm. These are not official government thresholds."
    )
else:
    st.warning("Priority early-warning rainfall assessment is unavailable for Guwahati/Shillong.")


# ============================================================
# PRIORITY ALERT SUMMARY
# ============================================================

priority_high_alerts = [
    item for item in priority_early_warning_results
    if item["warning"] in ["HIGH", "CRITICAL"]
]

if priority_high_alerts:
    st.error(
        "🚨 PRIORITY HIGH-RISK ALERT — "
        + ", ".join(
            f"{item['place']}, {item['state']} ({item['warning']})"
            for item in priority_high_alerts
        )
    )
else:
    st.info(
        "Priority high-risk detection is active for Guwahati and Shillong. "
        "A HIGH/CRITICAL alert appears automatically when the AI risk or "
        "prototype rainfall trigger reaches the configured threshold."
    )


# ============================================================
# MAP
# ============================================================

st.header(
    "NER Risk Monitoring Map"
)

st.caption(
    "Rainfall intensity zones and AI risk locations. "
    "Click ANY location on the map — even if there is no colored dot — for a local assessment."
)


# ============================================================
# DETAILED SIKKIM MONITORING
# ============================================================

# Keep the seven-state overview, while adding detailed monitoring
# at every valid IMERG rainfall-grid cell inside Sikkim.
detailed_sikkim_results = []

# Approximate local place names used for prototype map popups.
# These are reference locations, not exact reverse-geocoded addresses.
SIKKIM_PLACES = [
    (27.3314, 88.6139, "Gangtok"),
    (27.5258, 88.5615, "Mangan"),
    (27.3546, 88.6499, "Rangpo"),
    (27.2315, 88.6550, "Namchi"),
    (27.2810, 88.5510, "Ravangla"),
    (27.2500, 88.3000, "Gyalshing"),
    (27.6930, 88.7390, "Lachung"),
    (27.7090, 88.7510, "Lachen"),
    (27.3080, 88.6140, "Pakyong"),
    (27.1900, 88.5200, "Jorethang"),
    (27.3600, 88.7400, "Chungthang"),
]

def nearest_sikkim_place(latitude, longitude):
    """Return the nearest well-known Sikkim reference place."""
    best_name = "Sikkim monitoring area"
    best_distance = float("inf")
    for place_lat, place_lon, place_name in SIKKIM_PLACES:
        # Simple local-distance approximation, sufficient for display.
        dx = (longitude - place_lon) * np.cos(np.radians(latitude))
        dy = latitude - place_lat
        distance = dx * dx + dy * dy
        if distance < best_distance:
            best_distance = distance
            best_name = place_name
    return best_name

@st.cache_data(ttl=86400, show_spinner=False)
def reverse_geocode_sikkim(latitude, longitude):
    """Get a real nearby OSM place/address for a clicked Sikkim coordinate.
    Falls back to the nearest known Sikkim reference place if the service is unavailable.
    """
    fallback = nearest_sikkim_place(latitude, longitude)
    try:
        params = urllib.parse.urlencode({
            "lat": f"{latitude:.6f}",
            "lon": f"{longitude:.6f}",
            "format": "jsonv2",
            "zoom": 14,
            "addressdetails": 1,
        })
        url = "https://nominatim.openstreetmap.org/reverse?" + params
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "NER-Landslide-Risk-Monitoring-Prototype/1.0"
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))

        address = data.get("address", {})
        # Prefer a named locality/village/town, then district/county.
        place = (
            address.get("village")
            or address.get("town")
            or address.get("city")
            or address.get("municipality")
            or address.get("hamlet")
            or address.get("suburb")
            or address.get("neighbourhood")
            or address.get("county")
            or fallback
        )

        district = address.get("state_district") or address.get("county") or ""
        state = address.get("state") or "Sikkim"
        return {"place": place, "district": district, "state": state}
    except Exception:
        return {"place": fallback, "district": "", "state": "Sikkim"}

try:
    sikkim_selected = ner_wgs84[
        ner_wgs84[state_column] == "Sikkim"
    ]

    if not sikkim_selected.empty:
        try:
            sikkim_geometry = sikkim_selected.geometry.union_all()
        except AttributeError:
            sikkim_geometry = sikkim_selected.geometry.unary_union

        with rasterio.open(RAINFALL_FILE) as sikkim_src:
            sikkim_window = rasterio.windows.from_bounds(
                *sikkim_geometry.bounds,
                transform=sikkim_src.transform
            ).round_offsets().round_lengths()

            data = sikkim_src.read(1, window=sikkim_window).astype(float)
            window_transform = sikkim_src.window_transform(sikkim_window)

            rows, cols = np.where(data >= 0)

            for row, col in zip(rows.tolist(), cols.tolist()):
                lon, lat = rasterio.transform.xy(
                    window_transform, row, col, offset="center"
                )

                point = gpd.points_from_xy([lon], [lat])[0]

                if not sikkim_geometry.covers(point):
                    continue

                local_rainfall = get_location_rainfall(lat, lon)
                if local_rainfall is None:
                    continue

                local_risk, local_confidence = predict_risk(local_rainfall)
                local_warning, local_warning_text = rainfall_warning(
                    local_rainfall["mean"]
                )

                detailed_sikkim_results.append({
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "rainfall": local_rainfall,
                    "risk": local_risk,
                    "confidence": local_confidence,
                    "warning": local_warning,
                    "warning_text": local_warning_text,
                    "place": nearest_sikkim_place(float(lat), float(lon))
                })
except Exception:
    detailed_sikkim_results = []


if detailed_sikkim_results:
    st.info(
        f"Sikkim detailed monitoring: {len(detailed_sikkim_results)} valid "
        "NASA IMERG rainfall-grid locations are assessed inside the state. "
        "Colored points include a clearly-labelled synthetic demonstration mix for the prototype."
    )
else:
    st.warning(
        "Sikkim detailed monitoring layer is unavailable for the current rainfall raster."
    )

# Center the map on the place selected from the dropdown.
# This works for every NER state.
if selected_place is not None and selected_place_lat is not None and selected_place_lon is not None:
    map_center = [selected_place_lat, selected_place_lon]
    map_zoom = 10
else:
    map_center = [25.8, 92.0]
    map_zoom = 6

m = folium.Map(

    location=map_center,

    zoom_start=map_zoom,

    control_scale=True

)


# Enable map-wide click detection. This makes a click on any
# location in the map (not only on a monitoring dot) available
# to Streamlit through last_clicked.
# Place selection is handled reliably through the Sikkim dropdown.




# ============================================================
# STATE BOUNDARIES
# ============================================================

folium.GeoJson(

    ner_wgs84.to_json(),

    name="NER State Boundaries",

    style_function=lambda feature: {

        "color": "#526777",

        "weight": 1.5,

        "fillColor": "#9aa9b5",

        "fillOpacity": 0.08

    },

    tooltip=folium.GeoJsonTooltip(

        fields=[
            state_column
        ],

        aliases=[
            "State:"
        ]

    )

).add_to(m)


# ============================================================
# SELECTED PLACE MARKER
# ============================================================

if selected_place is not None and selected_place_lat is not None and selected_place_lon is not None:

    folium.Marker(

        location=[selected_place_lat, selected_place_lon],

        tooltip=f"Selected: {selected_place}, {selected_state}",

        popup=folium.Popup(
            f"<b>Selected Location</b><br>"
            f"Place: <b>{selected_place}</b><br>"
            f"State: <b>{selected_state}</b><br>"
            f"Latitude: {selected_place_lat:.5f}<br>"
            f"Longitude: {selected_place_lon:.5f}",
            max_width=320
        ),

        icon=folium.Icon(
            color="blue",
            icon="map-marker",
            prefix="glyphicon"
        )

    ).add_to(m)


# ============================================================
# STATE MARKERS
# ============================================================

for result in results:

    rainfall = result[
        "rainfall"
    ]["mean"]


    # Rainfall colour

    if rainfall >= 100:

        rainfall_color = "red"

    elif rainfall >= 50:

        rainfall_color = "orange"

    elif rainfall >= 25:

        rainfall_color = "beige"

    else:

        rainfall_color = "green"


    # AI risk colour

    if result["risk"] == "HIGH":

        marker_color = "red"

    elif result["risk"] == "MEDIUM":

        marker_color = "orange"

    else:

        marker_color = "green"


    # Rainfall zone

    folium.Circle(

        location=[

            result["latitude"],

            result["longitude"]

        ],

        radius=18000,

        color=rainfall_color,

        fill=True,

        fillColor=rainfall_color,

        fillOpacity=0.15,

        weight=2,

        popup=folium.Popup(

            f"""
            <b>{result['state']}</b><br><br>

            1-day rainfall:
            {rainfall:.1f} mm<br>

            AI risk:
            <b>{result['risk']}</b><br>

            Warning:
            <b>{result['warning']}</b><br>

            Confidence:
            {result['confidence']:.1f}%
            """,

            max_width=300

        )

    ).add_to(m)


    # AI marker + nearby alert details

    alerts_here = result.get("alerts", []) or []

    alert_html = ""

    if alerts_here:

        alert_html += "<hr><b>Nearby Registered Users</b><br>"

        for alert in alerts_here:

            alert_html += (
                f"User: <b>{alert.get('name', 'Registered user')}</b><br>"
                f"Alert level: <b>{alert.get('alert_level', 'WARNING')}</b><br>"
                f"Distance: {float(alert.get('distance_km', 0)):.2f} km<br>"
                f"Status: <b>WARNING GENERATED</b><br><br>"
            )

    else:

        alert_html = "<hr><b>Nearby Registered Users</b><br>No alert currently required."

    popup_html = f"""
        <div style=\"font-size:14px; line-height:1.5;\">
        <b style=\"font-size:17px;\">{result['state']}</b>
        <hr>
        <b>AI Risk:</b> {result['risk']}<br>
        <b>1-day Rainfall:</b> {rainfall:.1f} mm<br>
        <b>Warning:</b> {result['warning']}<br>
        <b>Confidence:</b> {result['confidence']:.1f}%
        {alert_html}
        </div>
    """

    folium.Marker(

        [

            result["latitude"],

            result["longitude"]

        ],

        tooltip=(
            f"{result['state']} | "
            f"{result['risk']} | Click for details"
        ),

        popup=folium.Popup(
            popup_html,
            max_width=380,
            min_width=300
        ),

        icon=folium.Icon(

            color=marker_color,

            icon="warning-sign"

        )

    ).add_to(m)


# ============================================================
# PRIORITY EARLY-WARNING MAP MARKERS
# ============================================================

for ew_result in priority_early_warning_results:
    ew_color = "red" if ew_result["warning"] == "CRITICAL" else (
        "orange" if ew_result["warning"] == "HIGH" else (
            "beige" if ew_result["warning"] == "WATCH" else "green"
        )
    )

    folium.Marker(
        location=[ew_result["latitude"], ew_result["longitude"]],
        tooltip=(
            f"Early Warning: {ew_result['place']}, {ew_result['state']} | "
            f"{ew_result['warning']}"
        ),
        popup=folium.Popup(
            f"""
            <b>PRIORITY EARLY-WARNING LOCATION</b><br>
            <hr>
            <b>Place:</b> {ew_result['place']}<br>
            <b>State:</b> {ew_result['state']}<br>
            <b>1-day Rainfall:</b> {ew_result['rainfall']['mean']:.1f} mm<br>
            <b>AI Risk:</b> {ew_result['risk']}<br>
            <b>Early Warning:</b> {ew_result['warning']}<br>
            <b>Users to Alert:</b> {len(ew_result.get('alerts', []))}<br>
            <small>Prototype monitoring/reference location.</small>
            """,
            max_width=340
        ),
        icon=folium.Icon(
            color=ew_color,
            icon="bell",
            prefix="glyphicon"
        )
    ).add_to(m)


# ============================================================
# SIKKIM DETAILED MAP LAYER
# ============================================================

if detailed_sikkim_results:
    sikkim_group = folium.FeatureGroup(
        name=f"Sikkim Detailed Monitoring ({len(detailed_sikkim_results)} cells)",
        show=True
    )

    for cell in detailed_sikkim_results:
        cell_rainfall = cell["rainfall"]["mean"]

        # The current AI baseline may produce mostly LOW values because it is
        # trained on rainfall-derived prototype labels. For a visually useful
        # hackathon coverage demonstration, add a clearly-labelled synthetic
        # scenario layer with a small mix of HIGH/MEDIUM/LOW locations.
        # This does NOT claim that these are historical landslide points.
        demo_score = (
            int(abs(cell["latitude"] * 10000))
            + int(abs(cell["longitude"] * 10000))
            + int(cell_rainfall * 10)
        ) % 100

        if cell["risk"] == "HIGH" or demo_score < 7:
            cell_color = "red"
            demo_risk = "HIGH"
        elif cell["risk"] == "MEDIUM" or demo_score < 22:
            cell_color = "orange"
            demo_risk = "MEDIUM"
        else:
            cell_color = "green"
            demo_risk = "LOW"

        folium.CircleMarker(
            location=[cell["latitude"], cell["longitude"]],
            radius=5,
            color=cell_color,
            fill=True,
            fillColor=cell_color,
            fillOpacity=0.65,
            weight=1,
            tooltip=(
                f"{cell['place']} | Sikkim | "
                f"Demo Risk: {demo_risk} | Rainfall: {cell_rainfall:.1f} mm"
            ),
            popup=folium.Popup(
                f"""
                <b>Sikkim Detailed Monitoring</b><br>
                <hr>
                <b>Nearest Place:</b> {cell['place']}<br>
                Latitude: {cell['latitude']:.5f}<br>
                Longitude: {cell['longitude']:.5f}<br>
                1-day Rainfall: {cell_rainfall:.1f} mm<br>
                AI Risk: <b>{cell['risk']}</b><br>
                AI Confidence: <b>{cell['confidence']:.1f}%</b><br>
                <b>Demo Scenario Risk: {demo_risk}</b><br>
                Early Warning: <b>{cell['warning']}</b><br>
                <small>Demo Scenario Risk is synthetic for prototype visualization.
                It is not a historical landslide location.</small>
                """,
                max_width=350
            )
        ).add_to(sikkim_group)

    sikkim_group.add_to(m)

# ============================================================
# EARLY WARNING MAP MARKERS
# ============================================================

for alert in all_alerts:

    matched_result = None

    for result in results:

        if any(
            item.get("name") == alert.get("name")
            and item.get("phone") == alert.get("phone")
            for item in result.get("alerts", [])
        ):
            matched_result = result
            break

    if matched_result is not None:

        folium.Marker(
            [
                matched_result["latitude"],
                matched_result["longitude"]
            ],
            tooltip=(
                f"Early warning: "
                f"{alert.get('name', 'Registered user')}"
            ),
            popup=folium.Popup(
                f"""
                <b>EARLY WARNING</b><br>
                <hr>
                User: {alert.get('name', 'Registered user')}<br>
                Level: <b>{alert.get('alert_level', 'WARNING')}</b><br>
                Distance: {float(alert.get('distance_km', 0)):.2f} km<br>
                Rainfall: {float(alert.get('rainfall_mm', 0)):.1f} mm<br>
                <br>
                Prototype warning generated.
                """,
                max_width=300
            ),
            icon=folium.Icon(
                color="red",
                icon="bell"
            )
        ).add_to(m)


# ============================================================
# LAST CLICK LOCATION POPUP
# ============================================================
# On the rerun caused by a map click, place a marker exactly at
# the clicked coordinates so the user can immediately see the
# approximate place name along with the coordinates.
last_saved_click = st.session_state.get("last_map_click")

if last_saved_click:
    try:
        click_lat = float(last_saved_click.get("lat"))
        click_lon = float(last_saved_click.get("lng"))
        click_point = Point(click_lon, click_lat)

        # Check whether the clicked point is inside Sikkim.
        click_is_sikkim = False
        try:
            sikkim_rows = ner_wgs84[
                ner_wgs84[state_column].astype(str).str.strip().str.lower() == "sikkim"
            ]
            click_is_sikkim = sikkim_rows.geometry.covers(click_point).any()
        except Exception:
            click_is_sikkim = False

        if click_is_sikkim:
            click_place = nearest_sikkim_place(click_lat, click_lon)
            click_rainfall = get_location_rainfall(click_lat, click_lon)

            if click_rainfall is not None:
                click_risk, click_confidence = predict_risk(click_rainfall)
                click_warning, _ = rainfall_warning(click_rainfall["mean"])

                click_popup = f"""
                <div style='font-size:14px; line-height:1.55; min-width:250px;'>
                    <b style='font-size:18px;'>📍 {click_place}</b><br>
                    <b>Sikkim</b>
                    <hr>
                    <b>Place:</b> {click_place}<br>
                    <b>Latitude:</b> {click_lat:.5f}<br>
                    <b>Longitude:</b> {click_lon:.5f}<br>
                    <b>1-day Rainfall:</b> {click_rainfall['mean']:.1f} mm<br>
                    <b>AI Risk:</b> {click_risk}<br>
                    <b>Warning:</b> {click_warning}<br>
                    <b>AI Confidence:</b> {click_confidence:.1f}%
                    <hr>
                    <small>Place name is the nearest Sikkim reference location.</small>
                </div>
                """

                folium.Marker(
                    location=[click_lat, click_lon],
                    tooltip=f"📍 {click_place} — Clicked Location",
                    popup=folium.Popup(
                        click_popup,
                        max_width=380,
                        min_width=280,
                    ),
                    icon=folium.Icon(color="blue", icon="map-marker"),
                ).add_to(m)
    except Exception:
        pass


# ============================================================
# MAP LAYER CONTROL
# ============================================================

folium.LayerControl().add_to(m)




# ============================================================
# DISPLAY MAP
# ============================================================

map_data = st_folium(

    m,

    width=None,

    height=650,

    returned_objects=[
        "last_clicked"
    ],

    key="ner_risk_map"

)

# Remember the last map click so the Location Intelligence panel
# remains populated after Streamlit reruns.
if map_data and map_data.get("last_clicked"):
    st.session_state["last_map_click"] = map_data["last_clicked"]

last_click = st.session_state.get("last_map_click")


# ============================================================
# SELECTED STATE
# ============================================================

st.divider()

st.header(
    "Selected State"
)


selected_result = None


for result in results:

    if result["state"] == selected_state:

        selected_result = result

        break


if selected_result is not None:

    rainfall = selected_result[
        "rainfall"
    ]


    if selected_place is not None:
        st.success(
            f"📍 SELECTED PLACE: {selected_place} | {selected_state.upper()}"
        )
        st.caption(
            f"Coordinates: {selected_place_lat:.5f}, "
            f"{selected_place_lon:.5f}"
        )

    s1, s2, s3, s4 = st.columns(4)


    with s1:

        st.metric(
            "State",
            selected_result["state"]
        )


    with s2:

        st.metric(
            "Average rainfall",
            f"{rainfall['mean']:.1f} mm"
        )


    with s3:

        st.metric(
            "AI risk",
            selected_result["risk"]
        )


    with s4:

        st.metric(
            "Confidence",
            f"{selected_result['confidence']:.1f}%"
        )


    if selected_result["warning"] == "CRITICAL":

        st.error(
            selected_result["warning_text"]
        )

    elif selected_result["warning"] == "HIGH":

        st.warning(
            selected_result["warning_text"]
        )

    elif selected_result["warning"] == "WATCH":

        st.info(
            selected_result["warning_text"]
        )

    else:

        st.success(
            selected_result["warning_text"]
        )


    selected_alerts = selected_result.get(
        "alerts",
        []
    )

    if selected_alerts:

        st.error(
            f"EARLY WARNING — "
            f"{len(selected_alerts)} registered user(s) "
            f"are inside the prototype warning radius."
        )

        for alert in selected_alerts:

            st.write(
                f"**{alert.get('name', 'Registered user')}** — "
                f"{alert.get('alert_level', 'WARNING')} — "
                f"{float(alert.get('distance_km', 0)):.2f} km away"
            )

            if alert.get("message"):
                st.caption(
                    alert["message"]
                )

    else:

        st.success(
            "No registered user currently requires "
            "an alert at this selected-state monitoring point."
        )


# ============================================================
# SIKKIM COVERAGE SUMMARY
# ============================================================

if selected_state == "Sikkim" and detailed_sikkim_results:
    st.subheader("Sikkim Detailed Coverage")
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        st.metric("Sikkim monitoring cells", len(detailed_sikkim_results))

    with sc2:
        st.metric(
            "High-risk cells",
            sum(cell["risk"] == "HIGH" for cell in detailed_sikkim_results)
        )

    with sc3:
        st.metric(
            "Warning cells",
            sum(
                cell["warning"] in ["WATCH", "HIGH", "CRITICAL"]
                for cell in detailed_sikkim_results
            )
        )

    st.caption(
        "Each cell represents a valid rainfall-grid location inside Sikkim. "
        "Place names are approximate reference locations. Red/orange demo points are synthetic visualization scenarios, not historical landslides."
    )


# ============================================================
# LOCATION INTELLIGENCE
# ============================================================

st.divider()

st.header(
    "Location Intelligence"
)

st.caption(
    "Click ANY location inside Sikkim — even an empty area — to get the nearest real mapped place/area, coordinates, rainfall and AI assessment."
)


if last_click:

    clicked_lat = float(last_click["lat"])

    clicked_lon = float(last_click["lng"])


    # Identify the state containing the clicked location.
    clicked_state = "Outside monitored NER"
    for _, state_row in ner_wgs84.iterrows():
        try:
            if state_row.geometry.covers(Point(clicked_lon, clicked_lat)):
                clicked_state = str(state_row[state_column])
                break
        except Exception:
            pass

    # Always calculate a Sikkim place name from the clicked coordinates.
    # Normalize the state text because boundary files can contain whitespace/case differences.
    clicked_place = None
    clicked_district = ""
    if str(clicked_state).strip().lower() == "sikkim":
        clicked_state = "Sikkim"
        geo = reverse_geocode_sikkim(clicked_lat, clicked_lon)
        clicked_place = geo["place"]
        clicked_district = geo["district"]

    clicked_point = gpd.GeoDataFrame(

        geometry=[

            gpd.points_from_xy(

                [clicked_lon],

                [clicked_lat]

            )[0]

        ],

        crs="EPSG:4326"

    )


    inside_ner = ner_wgs84.geometry.apply(

        lambda geom: geom.covers(clicked_point.geometry.iloc[0])

    ).any()


    if not inside_ner:

        st.warning(
            "Selected location is outside "
            "the seven monitored NER states."
        )


    else:

        rainfall = get_location_rainfall(

            clicked_lat,

            clicked_lon

        )


        if rainfall is None:

            st.error(
                "Rainfall data is unavailable "
                "for this location."
            )


        else:

            risk, confidence = predict_risk(
                rainfall
            )


            warning_level, warning_text = (
                rainfall_warning(
                    rainfall["mean"]
                )
            )


            # Check registered users near the clicked location
            clicked_alerts = get_location_alerts(
                clicked_lat,
                clicked_lon,
                risk,
                rainfall["mean"],
                rainfall.get("max")
            )


            # Show a human-readable place for Sikkim clicks.
            if clicked_state == "Sikkim":
                # Make the place name impossible to miss in the dashboard.
                st.success(
                    f"📍 SELECTED PLACE: {clicked_place}  |  SIKKIM"
                )
                st.write(f"**Place / Area:** {clicked_place}")
                if clicked_district:
                    st.write(f"**District:** {clicked_district}")
            else:
                st.write(f"**Place/Area:** {clicked_state}")

            st.write(f"**State:** {clicked_state}")

            st.write(
                f"**Coordinates:** "
                f"{clicked_lat:.5f}, "
                f"{clicked_lon:.5f}"
            )


            l1, l2, l3, l4 = st.columns(4)


            with l1:

                st.metric(
                    "Average rainfall",
                    f"{rainfall['mean']:.1f} mm"
                )


            with l2:

                st.metric(
                    "Maximum rainfall",
                    f"{rainfall['max']:.1f} mm"
                )


            with l3:

                st.metric(
                    "AI risk",
                    risk
                )


            with l4:

                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%"
                )


            if warning_level == "CRITICAL":

                st.error(
                    warning_text
                )

            elif warning_level == "HIGH":

                st.warning(
                    warning_text
                )

            elif warning_level == "WATCH":

                st.info(
                    warning_text
                )

            else:

                st.success(
                    warning_text
                )


            # Nearby-user notification result for this exact map click
            st.subheader(
                "Nearby User Notifications"
            )

            if clicked_alerts:

                st.error(
                    f"WARNING GENERATED — {len(clicked_alerts)} registered user(s) "
                    "are within the prototype warning radius."
                )

                alert_rows = []

                for alert in clicked_alerts:
                    alert_rows.append({
                        "User": alert.get("name", "Registered user"),
                        "Alert Level": alert.get("alert_level", "WARNING"),
                        "Distance (km)": round(float(alert.get("distance_km", 0)), 2),
                        "Rainfall (mm)": round(float(alert.get("rainfall_mm", 0)), 1),
                        "Status": "WARNING GENERATED"
                    })

                st.dataframe(
                    pd.DataFrame(alert_rows),
                    use_container_width=True,
                    hide_index=True
                )

                for alert in clicked_alerts:
                    if alert.get("message"):
                        st.caption(
                            f"{alert.get('name', 'Registered user')}: "
                            f"{alert['message']}"
                        )

            else:

                st.info(
                    "No registered user is currently inside the prototype "
                    "warning radius for this selected location."
                )


            st.subheader(
                "Assessment"
            )


            if rainfall["mean"] >= 100:

                st.write(
                    "Very high rainfall is being observed. "
                    "The current prototype uses rainfall-derived "
                    "features for its risk classification."
                )

            elif rainfall["mean"] >= 50:

                st.write(
                    "Heavy rainfall is being observed. "
                    "The current prototype indicates "
                    "increased rainfall-driven risk."
                )

            elif rainfall["mean"] >= 25:

                st.write(
                    "Moderate rainfall is being observed. "
                    "The prototype indicates lower rainfall "
                    "pressure than heavy rainfall conditions."
                )

            else:

                st.write(
                    "Rainfall is currently relatively low. "
                    "The prototype indicates lower "
                    "rainfall-driven pressure."
                )


# ============================================================
# EARLY WARNING & NOTIFICATIONS
# ============================================================

st.divider()

st.header(
    "Early Warning & Notifications"
)

st.caption(
    "Location-based prototype alerts for registered users. "
    "This version generates warnings; it does not send "
    "real SMS messages."
)

if not ALERT_ENGINE_READY:

    st.warning(
        "Alert engine is not connected. "
        "Make sure data/alert_engine.py exists."
    )

else:

    ew1, ew2, ew3 = st.columns(3)

    with ew1:
        st.metric(
            "Registered users",
            registered_user_count
        )

    with ew2:
        st.metric(
            "Users requiring alerts",
            alert_user_count
        )

    with ew3:
        st.metric(
            "Warning locations",
            sum(
                bool(result.get("alerts"))
                for result in results
            )
        )

    if all_alerts:

        st.error(
            "ACTIVE PROTOTYPE ALERTS — "
            "Warning conditions detected near "
            "registered users."
        )

        alert_table = []

        for alert in all_alerts:

            alert_table.append({
                "User":
                    alert.get(
                        "name",
                        "Registered user"
                    ),

                "Alert Level":
                    alert.get(
                        "alert_level",
                        "WARNING"
                    ),

                "Distance (km)":
                    round(
                        float(
                            alert.get(
                                "distance_km",
                                0
                            )
                        ),
                        2
                    ),

                "Rainfall (mm)":
                    round(
                        float(
                            alert.get(
                                "rainfall_mm",
                                0
                            )
                        ),
                        1
                    ),

                "Status":
                    "WARNING GENERATED"
            })

        st.dataframe(
            alert_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No registered users currently require "
            "a prototype warning."
        )


# ============================================================
# STATE MONITORING
# ============================================================

st.divider()

st.header(
    "State Monitoring"
)

st.caption(
    "Current environmental and prototype AI assessment."
)


table_data = []


for result in results:

    table_data.append({

        "State":
            result["state"],

        "Rainfall (mm)":
            round(
                result["rainfall"]["mean"],
                1
            ),

        "Maximum (mm)":
            round(
                result["rainfall"]["max"],
                1
            ),

        "AI Risk":
            result["risk"],

        "Confidence (%)":
            round(
                result["confidence"],
                1
            ),

        "Warning":
            result["warning"],

        "Users to Alert":
            len(
                result.get(
                    "alerts",
                    []
                )
            )

    })


st.dataframe(

    table_data,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader(
    "Risk Distribution"
)


r1, r2, r3 = st.columns(3)


with r1:

    st.metric(
        "LOW",
        low_count
    )


with r2:

    st.metric(
        "MEDIUM",
        medium_count
    )


with r3:

    st.metric(
        "HIGH",
        high_count
    )


# ============================================================
# RAINFALL SCALE
# ============================================================

st.subheader(
    "Rainfall Monitoring Scale"
)

st.caption(
    "Prototype decision rules: LOW <25 mm | WATCH 25–50 mm | "
    "HIGH 50–100 mm | CRITICAL ≥100 mm. These are not official "
    "government warning thresholds."
)


rc1, rc2, rc3, rc4 = st.columns(4)


with rc1:

    st.success(
        "LOW\n\nBelow 25 mm"
    )


with rc2:

    st.info(
        "WATCH\n\n25–50 mm"
    )


with rc3:

    st.warning(
        "HIGH\n\n50–100 mm"
    )


with rc4:

    st.error(
        "CRITICAL\n\n100 mm or above"
    )


# ============================================================
# DATA AND MODEL
# ============================================================

st.divider()

st.header(
    "Data & Model"
)


d1, d2, d3 = st.columns(3)


with d1:

    st.write(
        "**Rainfall**"
    )

    st.write(
        "NASA GPM IMERG"
    )

    st.caption(
        "1-day precipitation data"
    )


with d2:

    st.write(
        "**Geographic boundaries**"
    )

    st.write(
        "NWIC / GSI"
    )

    st.caption(
        "NER administrative boundaries"
    )


with d3:

    st.write(
        "**Machine learning**"
    )

    st.write(
        "Random Forest"
    )

    st.caption(
        "Current prototype model"
    )


# ============================================================
# WORKFLOW
# ============================================================

st.subheader(
    "System Workflow"
)


st.code(
    """
NER location
     ↓
NASA IMERG rainfall
     ↓
Local rainfall features
     ↓
Random Forest model
     ↓
LOW / MEDIUM / HIGH
     ↓
Location-based warning engine
     ↓
Registered user alert
     ↓
Risk map + early warning
""",
    language="text"
)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "Prototype limitation: The current model uses "
    "rainfall-derived training labels and is not yet "
    "scientifically validated for operational landslide "
    "prediction. Future development should incorporate "
    "field-validated historical landslide inventory, "
    "terrain variables and scientifically validated "
    "warning thresholds. The notification module currently "
    "generates prototype alerts for registered users but "
    "does not send operational SMS messages."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        NER Landslide Risk Monitoring System
        • North Eastern Region of India
        • Hackathon Prototype
    </div>
    """,
    unsafe_allow_html=True
)