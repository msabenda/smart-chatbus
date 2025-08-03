import joblib
from sklearn.preprocessing import LabelEncoder

# Re-create encoders with scikit-learn 1.7.0
day_encoder = LabelEncoder()
day_encoder.fit(['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'])
joblib.dump(day_encoder, 'day_encoder.pkl')

weather_encoder = LabelEncoder()
weather_encoder.fit(['Rainy', 'Sunny', 'Cloudy'])  # Include Cloudy as per training code
joblib.dump(weather_encoder, 'weather_encoder.pkl')

peak_hours_encoder = LabelEncoder()
peak_hours_encoder.fit(['No', 'Yes'])
joblib.dump(peak_hours_encoder, 'peak_hours_encoder.pkl')

weekends_encoder = LabelEncoder()
weekends_encoder.fit(['No', 'Yes'])
joblib.dump(weekends_encoder, 'weekends_encoder.pkl')

holidays_encoder = LabelEncoder()
holidays_encoder.fit(['No', 'Yes'])
joblib.dump(holidays_encoder, 'holidays_encoder.pkl')

print("Encoders re-saved successfully with scikit-learn 1.7.0.")