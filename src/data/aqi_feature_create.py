from src.utils.paths import PROCESSED_DIR

import os

def calculate_aqi(sub_indices, concentrations):
    # sub_indices: dictionary containing pollutant sub-index values
    # concentrations: dictionary containing pollutant concentrations

    # Check valid pollutants (non-zero values)
    valid_pollutants = sum(1 for value in concentrations.values() if value > 0)

    # Check PM10 or PM2.5 presence
    pm_present = (concentrations.get("PM10", 0) > 0 or concentrations.get("PM2.5", 0) > 0)

    if valid_pollutants < 3:
        return None, "At least 3 pollutants required"

    if not pm_present:
        return None, "Either PM10 or PM2.5 must be present"

    # AQI is maximum sub-index
    aqi = max(sub_indices.values())

    # AQI Categories
    if 0 <= aqi <= 50:
        category = "Good"
        impact = "Minimal Impact"

    elif 51 <= aqi <= 100:
        category = "Satisfactory"
        impact = "Minor breathing discomfort to sensitive people"

    elif 101 <= aqi <= 200:
        category = "Moderate"
        impact = ("Breathing discomfort to people with lung disease, heart disease, children and older adults")

    elif 201 <= aqi <= 300:
        category = "Poor"
        impact = "Breathing discomfort on prolonged exposure"

    elif 301 <= aqi <= 400:
        category = "Very Poor"
        impact = "Respiratory illness on prolonged exposure"

    else:
        category = "Severe"
        impact = "Respiratory effects even on healthy people"

    # return aqi, category, impact
    return aqi, category


concentrations = {
    "PM10": 121,
    "PM2.5": 34,
    "SO2": 0,
    "NO2": 8,
    "CO": 0,
    "O3": 57,
    "NH3": 34
}

sub_indices = {
    "PM10": 114,
    "PM2.5": 57,
    "SO2": 0,
    "NO2": 10,
    "CO": 0,
    "O3": 57,
    "NH3": 9
}

aqi, category, impact = calculate_aqi(sub_indices, concentrations)

print("=" * 50)
print("CPCB AIR QUALITY INDEX REPORT")
print("=" * 50)
print("Station :", "NSIT")
print("City    :", "Delhi")
print("State   :", "Delhi")
print("Date    :", datetime.now().strftime("%d-%m-%Y"))
print("-" * 50)
print("AQI     :", aqi)
print("Category:", category)
print("Impact  :", impact)
print("=" * 50)

if __name__ == "__main__":
    main()