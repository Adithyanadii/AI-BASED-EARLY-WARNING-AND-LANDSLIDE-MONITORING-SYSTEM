import streamlit as st
import joblib
import rasterio
import numpy as np
import geopandas as gpd
import folium

from streamlit_folium import st_folium


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

def rainfall_warning(
    rainfall
):

    if rainfall >= 100:

        return (
            "RED",
            "Severe rainfall warning"
        )

    elif rainfall >= 50:

        return (
            "ORANGE",
            "Heavy rainfall watch"
        )

    elif rainfall >= 25:

        return (
            "YELLOW",
            "Rainfall alert"
        )

    else:

        return (
            "GREEN",
            "Normal rainfall"
        )


# ============================================================
# EARLY WARNING SYSTEM
# ============================================================

def early_warning(rainfall):
    """
    Prototype early-warning classification based on 1-day rainfall.
    These thresholds are for the hackathon prototype and are NOT
    official government warning thresholds.
    """

    if rainfall >= 100:
        return (
            "CRITICAL",
            "Immediate attention recommended: very high 1-day rainfall detected."
        )
    elif rainfall >= 50:
        return (
            "HIGH",
            "High rainfall detected: monitor landslide-prone areas closely."
        )
    elif rainfall >= 25:
        return (
            "WATCH",
            "Rainfall is elevated: continue close monitoring."
        )
    else:
        return (
            "LOW",
            "Rainfall is currently below the prototype warning threshold."
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


    point = (
        selected
        .geometry
        .iloc[0]
        .representative_point()
    )


    latitude = float(
        point.y
    )

    longitude = float(
        point.x
    )


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

        "early_warning_level":
            early_warning(rainfall["mean"])[0],

        "early_warning_text":
            early_warning(rainfall["mean"])[1]

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
        "RED",
        "ORANGE"
    ]

    for result in results
)


critical_warning_count = sum(
    result["early_warning_level"] == "CRITICAL"
    for result in results
)

high_warning_count = sum(
    result["early_warning_level"] == "HIGH"
    for result in results
)

watch_warning_count = sum(
    result["early_warning_level"] == "WATCH"
    for result in results
)

low_warning_count = sum(
    result["early_warning_level"] == "LOW"
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
# REGIONAL OVERVIEW
# ============================================================

st.header(
    "Regional Overview"
)

st.caption(
    "Current monitoring summary across the seven NER states."
)


c1, c2, c3, c4 = st.columns(4)


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
# EARLY WARNING DASHBOARD
# ============================================================

st.subheader(
    "Early Warning Dashboard"
)

st.caption(
    "Prototype warning level derived from 1-day rainfall at monitored locations."
)

ew1, ew2, ew3, ew4 = st.columns(4)

with ew1:
    st.metric("🔴 Critical", critical_warning_count)

with ew2:
    st.metric("🟠 High", high_warning_count)

with ew3:
    st.metric("🟡 Watch", watch_warning_count)

with ew4:
    st.metric("🟢 Low", low_warning_count)

if critical_warning_count > 0:
    st.error(
        f"CRITICAL EARLY WARNING — {critical_warning_count} monitored "
        "location(s) have rainfall at or above 100 mm."
    )
elif high_warning_count > 0:
    st.warning(
        f"HIGH EARLY WARNING — {high_warning_count} monitored location(s) "
        "have rainfall at or above 50 mm."
    )
elif watch_warning_count > 0:
    st.info(
        f"WATCH — {watch_warning_count} monitored location(s) "
        "have rainfall at or above 25 mm."
    )
else:
    st.success(
        "LOW EARLY WARNING — No monitored representative location "
        "currently crosses the prototype warning threshold."
    )


# ============================================================
# MAP
# ============================================================

st.header(
    "NER Risk Monitoring Map"
)

st.caption(
    "Rainfall intensity zones and AI risk locations. "
    "Click a location inside NER for analysis."
)


m = folium.Map(

    location=[
        25.8,
        92.0
    ],

    zoom_start=6,

    control_scale=True

)


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

            Early warning:
            <b>{result['early_warning_level']}</b><br>

            Confidence:
            {result['confidence']:.1f}%
            """,

            max_width=300

        )

    ).add_to(m)


    # AI marker

    folium.Marker(

        [

            result["latitude"],

            result["longitude"]

        ],

        tooltip=(

            f"{result['state']} | "

            f"{result['risk']}"

        ),

        popup=folium.Popup(

            f"""
            <b>{result['state']}</b>

            <hr>

            AI Risk:
            <b>{result['risk']}</b>

            <br>

            Rainfall:
            {rainfall:.1f} mm

            <br>

            Warning:
            {result['warning']}

            <br>

            Early warning:
            <b>{result['early_warning_level']}</b>

            <br>

            Confidence:
            {result['confidence']:.1f}%

            """,

            max_width=300

        ),

        icon=folium.Icon(

            color=marker_color,

            icon="warning-sign"

        )

    ).add_to(m)


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
    ]

)


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


    s1, s2, s3, s4, s5 = st.columns(5)


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


    with s5:

        st.metric(
            "Early warning",
            selected_result["early_warning_level"]
        )


    if selected_result["warning"] == "RED":

        st.error(
            selected_result["warning_text"]
        )

    elif selected_result["warning"] == "ORANGE":

        st.warning(
            selected_result["warning_text"]
        )

    elif selected_result["warning"] == "YELLOW":

        st.info(
            selected_result["warning_text"]
        )

    else:

        st.success(
            selected_result["warning_text"]
        )


# ============================================================
# LOCATION INTELLIGENCE
# ============================================================

st.divider()

st.header(
    "Location Intelligence"
)

st.caption(
    "Click any location inside the NER map to "
    "perform a local rainfall and AI assessment."
)


if (

    map_data

    and

    map_data.get(
        "last_clicked"
    )

):

    clicked_lat = (
        map_data[
            "last_clicked"
        ]["lat"]
    )


    clicked_lon = (
        map_data[
            "last_clicked"
        ]["lng"]
    )


    clicked_point = gpd.GeoDataFrame(

        geometry=[

            gpd.points_from_xy(

                [clicked_lon],

                [clicked_lat]

            )[0]

        ],

        crs="EPSG:4326"

    )


    inside_ner = ner_wgs84.contains(

        clicked_point.geometry.iloc[0]

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


            st.write(
                f"**Coordinates:** "
                f"{clicked_lat:.5f}, "
                f"{clicked_lon:.5f}"
            )


            l1, l2, l3, l4, l5 = st.columns(5)


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


            clicked_early_warning, clicked_warning_text = early_warning(
                rainfall["mean"]
            )

            with l5:

                st.metric(
                    "Early warning",
                    clicked_early_warning
                )


            if warning_level == "RED":

                st.error(
                    warning_text
                )

            elif warning_level == "ORANGE":

                st.warning(
                    warning_text
                )

            elif warning_level == "YELLOW":

                st.info(
                    warning_text
                )

            else:

                st.success(
                    warning_text
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

        "Early Warning":
            result["early_warning_level"]

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


st.caption("Early warning distribution")

er1, er2, er3, er4 = st.columns(4)

with er1:
    st.metric("🔴 Critical", critical_warning_count)

with er2:
    st.metric("🟠 High", high_warning_count)

with er3:
    st.metric("🟡 Watch", watch_warning_count)

with er4:
    st.metric("🟢 Low", low_warning_count)


# ============================================================
# RAINFALL SCALE
# ============================================================

st.subheader(
    "Rainfall Monitoring Scale"
)


rc1, rc2, rc3, rc4 = st.columns(4)


with rc1:

    st.success(
        "🟢 LOW\n\nBelow 25 mm"
    )


with rc2:

    st.info(
        "🟡 WATCH / ALERT\n\n25–50 mm"
    )


with rc3:

    st.warning(
        "🟠 HIGH\n\n50–100 mm"
    )


with rc4:

    st.error(
        "🔴 CRITICAL\n\n100 mm or above"
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
Risk map + warning
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
    "warning thresholds."
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