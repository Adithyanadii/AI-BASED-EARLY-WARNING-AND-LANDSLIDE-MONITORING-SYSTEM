import pandas as pd
import math


# ============================================================
# CONFIGURATION
# ============================================================

USERS_FILE = "data/registered_users.csv"

# Prototype warning thresholds.
# These are NOT official government warning thresholds.
WATCH_RAINFALL_MM = 25
HIGH_RAINFALL_MM = 50
CRITICAL_RAINFALL_MM = 100

DEFAULT_RADIUS_KM = 25


# ============================================================
# DISTANCE CALCULATION
# ============================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two geographic coordinates
    using the Haversine formula.

    Returns distance in kilometres.
    """

    R = 6371.0

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    # Protect against very small floating-point errors.
    a = max(0.0, min(1.0, a))

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


# ============================================================
# RAINFALL SEVERITY
# ============================================================

def get_rainfall_level(rainfall):
    """
    Convert rainfall into a prototype severity level.

    These thresholds are demonstration thresholds,
    not official government warning thresholds.
    """

    rainfall = float(rainfall)

    if rainfall >= CRITICAL_RAINFALL_MM:
        return "CRITICAL"

    elif rainfall >= HIGH_RAINFALL_MM:
        return "HIGH"

    elif rainfall >= WATCH_RAINFALL_MM:
        return "WATCH"

    else:
        return "LOW"


# ============================================================
# AI RISK SCORE
# ============================================================

def get_risk_score(risk):
    """
    Convert the AI risk category into a numerical
    prototype score.

    LOW    = 0
    MEDIUM = 1
    HIGH   = 2
    """

    risk = str(risk).upper().strip()

    if risk == "HIGH":
        return 2

    elif risk == "MEDIUM":
        return 1

    else:
        return 0


# ============================================================
# EARLY WARNING DECISION
# ============================================================

def get_alert_level(risk, rainfall):
    """
    Combine AI risk and rainfall severity to determine
    the final prototype early-warning level.

    The system intentionally requires stronger evidence
    before generating a CRITICAL warning.
    """

    rainfall = float(rainfall)

    rainfall_level = get_rainfall_level(rainfall)
    risk_score = get_risk_score(risk)

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------
    # Very high rainfall automatically produces a critical
    # prototype warning.
    #
    # OR
    #
    # High AI risk combined with high rainfall.
    # --------------------------------------------------------

    if rainfall >= CRITICAL_RAINFALL_MM:
        return "CRITICAL"

    if risk_score == 2 and rainfall >= HIGH_RAINFALL_MM:
        return "CRITICAL"

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if rainfall_level == "HIGH":
        return "HIGH"

    if risk_score == 2:
        return "HIGH"

    if risk_score == 1 and rainfall >= WATCH_RAINFALL_MM:
        return "HIGH"

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    if rainfall_level == "WATCH":
        return "WATCH"

    if risk_score == 1:
        return "WATCH"

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return "LOW"


# ============================================================
# WARNING DESCRIPTION
# ============================================================

def get_warning_description(alert_level):
    """
    Human-readable explanation for the dashboard.
    """

    descriptions = {

        "LOW":
            "No immediate prototype warning condition detected.",

        "WATCH":
            "Elevated rainfall or AI risk detected. "
            "Continue monitoring local conditions.",

        "HIGH":
            "Significant rainfall and/or elevated AI risk detected. "
            "Enhanced monitoring is recommended.",

        "CRITICAL":
            "Very high rainfall and elevated risk conditions detected. "
            "Immediate attention and local authority guidance are recommended."
    }

    return descriptions.get(
        alert_level,
        "Warning condition detected."
    )


# ============================================================
# ALERT MESSAGE
# ============================================================

def create_alert_message(
    alert_level,
    risk,
    rainfall,
    distance_km
):
    """
    Generate a clear location-based warning message.
    """

    description = get_warning_description(alert_level)

    return (
        f"LANDSLIDE {alert_level} WARNING. "
        f"AI risk: {risk}. "
        f"Rainfall: {float(rainfall):.1f} mm. "
        f"Your location is approximately "
        f"{float(distance_km):.1f} km from the monitored risk location. "
        f"{description} "
        f"Please follow instructions from local authorities."
    )


# ============================================================
# CHECK USERS FOR ALERTS
# ============================================================

def check_users_for_alerts(
    risk_location_lat,
    risk_location_lon,
    risk,
    rainfall,
    radius_km=DEFAULT_RADIUS_KM
):
    """
    Find registered users within the warning radius.

    Alerts are generated only for:

        WATCH
        HIGH
        CRITICAL

    LOW conditions do not generate user alerts.
    """

    # --------------------------------------------------------
    # Load registered users
    # --------------------------------------------------------

    try:

        users = pd.read_csv(USERS_FILE)

    except FileNotFoundError:

        return []

    except Exception:

        return []

    # --------------------------------------------------------
    # Validate warning inputs
    # --------------------------------------------------------

    try:

        risk_location_lat = float(risk_location_lat)
        risk_location_lon = float(risk_location_lon)
        rainfall = float(rainfall)

    except (TypeError, ValueError):

        return []

    # --------------------------------------------------------
    # Determine alert level
    # --------------------------------------------------------

    alert_level = get_alert_level(
        risk,
        rainfall
    )

    # LOW = no notification
    if alert_level == "LOW":
        return []

    alerts = []

    # --------------------------------------------------------
    # Check every registered user
    # --------------------------------------------------------

    for _, user in users.iterrows():

        try:

            user_lat = float(user["latitude"])
            user_lon = float(user["longitude"])

        except (KeyError, TypeError, ValueError):

            continue

        distance = calculate_distance(
            risk_location_lat,
            risk_location_lon,
            user_lat,
            user_lon
        )

        # ----------------------------------------------------
        # User is inside warning radius
        # ----------------------------------------------------

        if distance <= radius_km:

            message = create_alert_message(
                alert_level,
                risk,
                rainfall,
                distance
            )

            alerts.append({

                "name":
                    str(
                        user.get(
                            "name",
                            "Registered user"
                        )
                    ),

                "phone":
                    str(
                        user.get(
                            "phone",
                            ""
                        )
                    ),

                "latitude":
                    user_lat,

                "longitude":
                    user_lon,

                "distance_km":
                    round(
                        distance,
                        2
                    ),

                "risk":
                    str(risk).upper(),

                "rainfall_mm":
                    round(
                        rainfall,
                        2
                    ),

                "rainfall_level":
                    get_rainfall_level(
                        rainfall
                    ),

                "alert_level":
                    alert_level,

                "message":
                    message,

                "status":
                    "WARNING GENERATED"
            })

    return alerts


# ============================================================
# TEST THE ENGINE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NER LANDSLIDE EARLY WARNING ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Test location
    # --------------------------------------------------------

    test_lat = 27.3389
    test_lon = 88.6065

    # --------------------------------------------------------
    # Test AI result
    # --------------------------------------------------------

    test_risk = "HIGH"

    # --------------------------------------------------------
    # Test rainfall
    # --------------------------------------------------------

    test_rainfall = 75

    # --------------------------------------------------------
    # Calculate warning
    # --------------------------------------------------------

    warning = get_alert_level(
        test_risk,
        test_rainfall
    )

    rainfall_level = get_rainfall_level(
        test_rainfall
    )

    print()
    print(f"AI Risk       : {test_risk}")
    print(f"Rainfall      : {test_rainfall} mm")
    print(f"Rainfall Level: {rainfall_level}")
    print(f"Warning Level : {warning}")

    # --------------------------------------------------------
    # Find users
    # --------------------------------------------------------

    alerts = check_users_for_alerts(
        test_lat,
        test_lon,
        test_risk,
        test_rainfall
    )

    print(
        f"Users requiring alerts: {len(alerts)}"
    )

    # --------------------------------------------------------
    # Display alerts
    # --------------------------------------------------------

    for alert in alerts:

        print()
        print("🚨 ALERT")
        print(
            f"User       : {alert['name']}"
        )
        print(
            f"Distance   : {alert['distance_km']} km"
        )
        print(
            f"AI Risk    : {alert['risk']}"
        )
        print(
            f"Rainfall   : {alert['rainfall_mm']} mm"
        )
        print(
            f"Rain Level : {alert['rainfall_level']}"
        )
        print(
            f"Alert      : {alert['alert_level']}"
        )
        print(
            f"Status     : {alert['status']}"
        )
        print(
            f"Message    : {alert['message']}"
        )

    print()
    print("=" * 60)