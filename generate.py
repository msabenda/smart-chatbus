import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define Tanzania-specific and global holidays for 2025
holidays = {
    "2025-01-01": "New Year's Day",
    "2025-01-12": "Zanzibar Revolution Day",
    "2025-04-07": "Sheikh Abeid Amani Karume Day",
    "2025-04-18": "Good Friday",
    "2025-04-20": "Easter Sunday",
    "2025-04-21": "Easter Monday",
    "2025-04-26": "Union Day",
    "2025-07-07": "Saba Saba",
    "2025-08-08": "Nane Nane",
    "2025-10-14": "Nyerere Day",
    "2025-12-09": "Independence Day",
    "2025-12-25": "Christmas Day",
    "2025-12-26": "Boxing Day",
    "2025-03-30": "Eid al-Fitr (estimated)",
    "2025-06-06": "Eid al-Adha (estimated)"
}

# Generate time slots (5 AM–10 PM, 15-min intervals)
def generate_time_slots():
    times = []
    for hour in range(5, 22):
        for minute in [0, 15, 30, 45]:
            times.append(f"{hour:02d}:{minute:02d}:00")
    return times

# Generate dataset
def generate_dataset(start_date, end_date, stations=2):
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    times = generate_time_slots()
    data = []
    
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        day = date.strftime("%A")
        is_weekend = day in ["Saturday", "Sunday"]
        is_holiday = date_str in holidays
        # Simulate rainy season (Mar–May, Nov–Dec)
        month = date.month
        rainy_prob = 0.5 if month in [3, 4, 5, 11, 12] else 0.2
        
        for time in times:
            hour = int(time[:2])
            is_peak = (6 <= hour < 9) or (16 <= hour < 19)
            weather = np.random.choice(["sunny", "rainy"], p=[1 - rainy_prob, rainy_prob])
            
            for station in range(1, stations + 1):
                # Base passenger count (mean=60, std=10)
                passengers = int(np.random.normal(60, 10))
                # Adjust for factors
                if is_peak:
                    passengers = int(passengers * 1.5)
                if is_weekend:
                    passengers = int(passengers * 0.8)
                if is_holiday:
                    passengers = int(passengers * 1.3)
                if weather == "rainy":
                    passengers = int(passengers * 0.85)
                # Clip to realistic range (10–150)
                passengers = max(10, min(150, passengers))
                
                data.append({
                    "date": date_str,
                    "day": day,
                    "time_value": time,
                    "passengers": passengers,
                    "weather": weather,
                    "peak_hours": is_peak,
                    "weekends": is_weekend,
                    "holidays": is_holiday
                })
    
    df = pd.DataFrame(data)
    # Downsample to ~8000 rows
    if len(df) > 8000:
        df = df.sample(n=8000, random_state=42)
    return df

# Set parameters
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 2, 28)  # 2 months
stations = 2  # 2 stations for ~8000 rows

# Generate and save dataset
df = generate_dataset(start_date, end_date, stations)
df.to_csv('dart_mwendokasi_realistic.csv', index=False)
print(f"Generated {len(df)} rows and saved to 'dart_mwendokasi_realistic.csv'")
print(df.head())