from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime, timedelta
import xgboost as xgb
import json
import re
import random

app = Flask(__name__)

# Enable CORS for all routes, allowing requests from any origin
CORS(app, resources={r"/*": {"origins": "*"}})

# Load XGBoost UBJSON model and encoders at startup (cached)
booster = xgb.Booster()
booster.load_model('xgboost_model.ubj')
model = xgb.XGBRegressor()
model._Booster = booster  # Attach booster manually

encoders = {
    'day': joblib.load('day_encoder.pkl'),
    'weather': joblib.load('weather_encoder.pkl'),
    'peak_hours': joblib.load('peak_hours_encoder.pkl'),
    'weekends': joblib.load('weekends_encoder.pkl'),
    'holidays': joblib.load('holidays_encoder.pkl')
}

# Use the feature order from the trained model
training_feature_order = ['day', 'weather', 'time_value', 'peak_hours', 'weekends', 'holidays', 'date_ordinal']

class NLPProcessor:
    """Enhanced Natural Language Processing layer with better understanding"""
    
    @staticmethod
    def detect_language(text):
        """Detect if the text is in Swahili or English"""
        swahili_keywords = [
            'je', 'saa', 'jumamosi', 'jumapili', 'jumatatu', 'jumanne', 'jumatano', 
            'alhamisi', 'ijumaa', 'abiria', 'basi', 'leo', 'kesho', 'jana',
            'mchana', 'usiku', 'asubuhi', 'jioni', 'mvua', 'jua', 'baridi',
            'ni', 'kuna', 'ngapi', 'idadi', 'watu', 'wengi', 'wachache',
            'ninaweza', 'naomba', 'tafadhali', 'samahani', 'karibu'
        ]
        
        text_lower = text.lower()
        swahili_count = sum(1 for word in swahili_keywords if word in text_lower)
        
        return 'Swahili' if swahili_count > 0 else 'English'
    
    @staticmethod
    def is_prediction_request(text):
        """Check if the text is asking for a passenger prediction"""
        text_lower = text.lower()
        
        # Prediction keywords in English
        english_keywords = [
            'passenger', 'passengers', 'people', 'crowd', 'crowded', 'busy',
            'how many', 'predict', 'forecast', 'expect', 'anticipate',
            'will there be', 'going to be', 'travel time', 'best time',
            'when to travel', 'avoid crowds', 'less crowded', 'peak', 'rush'
        ]
        
        # Prediction keywords in Swahili
        swahili_keywords = [
            'abiria', 'watu', 'idadi', 'ngapi', 'wengi', 'wachache',
            'msongamano', 'kujaa', 'tupu', 'wakati', 'bora', 'mzuri',
            'kusafiri', 'basi', 'gari', 'hatua'
        ]
        
        # Time/day indicators
        time_indicators = [
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'jumatatu', 'jumanne', 'jumatano', 'alhamisi', 'ijumaa', 'jumamosi', 'jumapili',
            'morning', 'afternoon', 'evening', 'night', 'asubuhi', 'mchana', 'jioni', 'usiku',
            'today', 'tomorrow', 'yesterday', 'leo', 'kesho', 'jana',
            'am', 'pm', 'o\'clock', 'saa', ':',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'
        ]
        
        # Check for prediction keywords
        has_prediction_keyword = any(keyword in text_lower for keyword in english_keywords + swahili_keywords)
        has_time_indicator = any(indicator in text_lower for indicator in time_indicators)
        
        return has_prediction_keyword or has_time_indicator
    
    @staticmethod
    def extract_time_from_text(text):
        """Extract time from natural language text"""
        text_lower = text.lower()
        
        # Time patterns
        time_patterns = [
            r'(\d{1,2}):(\d{2})\s*(am|pm)?',
            r'(\d{1,2})\s*(am|pm)',
            r'at\s+(\d{1,2}):(\d{2})',
            r'saa\s+(\d{1,2})',  # Swahili
            r'(\d{1,2})\s*o\'?clock'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2)) if len(match.groups()) > 1 and match.group(2) and match.group(2).isdigit() else 0
                
                # Handle AM/PM
                if len(match.groups()) > 2 and match.group(3):
                    if match.group(3).lower() == 'pm' and hour != 12:
                        hour += 12
                    elif match.group(3).lower() == 'am' and hour == 12:
                        hour = 0
                
                return f"{hour:02d}:{minute:02d}"
        
        # Named times
        named_times = {
            'morning': '08:00', 'afternoon': '14:00', 'evening': '18:00', 'night': '20:00',
            'noon': '12:00', 'midnight': '00:00', 'dawn': '06:00', 'dusk': '19:00',
            'asubuhi': '08:00', 'mchana': '12:00', 'jioni': '18:00', 'usiku': '20:00'
        }
        
        for time_word, time_value in named_times.items():
            if time_word in text_lower:
                return time_value
        
        return '08:00'  # Default
    
    @staticmethod
    def extract_day_from_text(text):
        """Extract day from natural language text"""
        text_lower = text.lower()
        
        # English days
        english_days = {
            'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday',
            'thursday': 'Thursday', 'friday': 'Friday', 'saturday': 'Saturday', 'sunday': 'Sunday'
        }
        
        # Swahili days
        swahili_days = {
            'jumatatu': 'Monday', 'jumanne': 'Tuesday', 'jumatano': 'Wednesday',
            'alhamisi': 'Thursday', 'ijumaa': 'Friday', 'jumamosi': 'Saturday', 'jumapili': 'Sunday'
        }
        
        # Check for specific days
        for eng_day, day_name in english_days.items():
            if eng_day in text_lower:
                return day_name
        
        for swah_day, day_name in swahili_days.items():
            if swah_day in text_lower:
                return day_name
        
        # Relative days
        today = datetime.now()
        if 'today' in text_lower or 'leo' in text_lower:
            return today.strftime('%A')
        elif 'tomorrow' in text_lower or 'kesho' in text_lower:
            return (today + timedelta(days=1)).strftime('%A')
        elif 'yesterday' in text_lower or 'jana' in text_lower:
            return (today - timedelta(days=1)).strftime('%A')
        
        return today.strftime('%A')  # Default to today
    
    @staticmethod
    def extract_weather_from_text(text):
        """Extract weather from natural language text"""
        text_lower = text.lower()
        
        weather_keywords = {
            'sunny': 'Sunny', 'rain': 'Rainy', 'rainy': 'Rainy', 'cloudy': 'Cloudy',
            'clear': 'Sunny', 'storm': 'Rainy', 'drizzle': 'Rainy',
            'mvua': 'Rainy', 'jua': 'Sunny', 'mawingo': 'Cloudy'
        }
        
        for keyword, weather in weather_keywords.items():
            if keyword in text_lower:
                return weather
        
        return 'Sunny'  # Default
    
    @staticmethod
    def extract_date_from_text(text):
        """Extract date from natural language text"""
        text_lower = text.lower()
        today = datetime.now()
        
        # Date patterns
        date_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(\d{1,2})-(\d{1,2})-(\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if pattern == date_patterns[0]:  # YYYY-MM-DD
                    return f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
                else:  # MM/DD/YYYY or MM-DD-YYYY
                    return f"{match.group(3)}-{match.group(1).zfill(2)}-{match.group(2).zfill(2)}"
        
        # Relative dates
        if 'today' in text_lower or 'leo' in text_lower:
            return today.strftime('%Y-%m-%d')
        elif 'tomorrow' in text_lower or 'kesho' in text_lower:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'yesterday' in text_lower or 'jana' in text_lower:
            return (today - timedelta(days=1)).strftime('%Y-%m-%d')
        
        return today.strftime('%Y-%m-%d')  # Default to today
    
    @staticmethod
    def is_peak_hours(time_str):
        """Determine if the time is during peak hours"""
        hour = int(time_str.split(':')[0])
        # Peak hours: 7-9 AM and 5-7 PM
        return 'Yes' if (7 <= hour <= 9) or (17 <= hour <= 19) else 'No'
    
    @staticmethod
    def is_weekend(day):
        """Check if the day is a weekend"""
        return 'Yes' if day in ['Saturday', 'Sunday'] else 'No'
    
    @staticmethod
    def is_holiday_from_text(text):
        """Check if text mentions holidays"""
        text_lower = text.lower()
        holiday_keywords = ['holiday', 'christmas', 'new year', 'easter', 'sikukuu']
        return 'Yes' if any(keyword in text_lower for keyword in holiday_keywords) else 'No'
    
    @classmethod
    def extract_structured_data(cls, prompt):
        """Main method to extract structured data from natural language prompt"""
        try:
            date_str = cls.extract_date_from_text(prompt)
            time_str = cls.extract_time_from_text(prompt)
            day = cls.extract_day_from_text(prompt)
            weather = cls.extract_weather_from_text(prompt)
            holidays = cls.is_holiday_from_text(prompt)
            
            # Calculate derived features
            peak_hours = cls.is_peak_hours(time_str)
            weekends = cls.is_weekend(day)
            
            return {
                "date": date_str,
                "time": time_str,
                "day": day,
                "weather": weather,
                "peak_hours": peak_hours,
                "weekends": weekends,
                "holidays": holidays
            }
        
        except Exception as e:
            print(f"Error extracting structured data: {e}")
            # Return safe defaults
            today = datetime.now()
            return {
                "date": today.strftime('%Y-%m-%d'),
                "time": "08:00",
                "day": today.strftime('%A'),
                "weather": "Sunny",
                "peak_hours": "No",
                "weekends": "Yes" if today.weekday() >= 5 else "No",
                "holidays": "No"
            }

class XGBoostPredictor:
    """XGBoost model layer for passenger predictions"""
    
    @staticmethod
    def predict_passengers(structured_data):
        """Predict passenger count from structured data"""
        try:
            # Convert structured data to model input format
            date_ordinal = pd.to_datetime(structured_data['date']).toordinal()
            
            # Convert time to minutes
            time_parts = structured_data['time'].split(':')
            time_minutes = int(time_parts[0]) * 60 + int(time_parts[1])
            
            # Create input dataframe with proper column order
            input_data = pd.DataFrame({
                'date_ordinal': [date_ordinal],
                'time_value': [time_minutes],
                'day': [encoders['day'].transform([structured_data['day']])[0]],
                'weather': [encoders['weather'].transform([structured_data['weather']])[0]],
                'peak_hours': [encoders['peak_hours'].transform([structured_data['peak_hours']])[0]],
                'weekends': [encoders['weekends'].transform([structured_data['weekends']])[0]],
                'holidays': [encoders['holidays'].transform([structured_data['holidays']])[0]]
            })[training_feature_order]  # Reorder columns to match training order
            
            prediction = model.predict(input_data)
            return int(round(prediction[0]))
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return 43  # Default fallback prediction

class ConversationalResponseGenerator:
    """Enhanced response generation with ChatGPT-like conversational style"""
    
    @staticmethod
    def get_crowding_insights(prediction, structured_data, language='English'):
        """Generate detailed insights about the prediction"""
        if language == 'Swahili':
            if prediction < 20:
                return {
                    'level': 'kidogo sana',
                    'description': 'Basi litakuwa tupu kabisa',
                    'advice': 'Wakati mzuri kabisa wa kusafiri. Utapata nafasi nyingi na utaweza kukaa bila shida.'
                }
            elif prediction < 35:
                return {
                    'level': 'kidogo',
                    'description': 'Abiria wachache',
                    'advice': 'Wakati mzuri wa kusafiri. Utapata nafasi ya kukaa na hewa safi.'
                }
            elif prediction < 50:
                return {
                    'level': 'wastani',
                    'description': 'Idadi ya kawaida ya abiria',
                    'advice': 'Kiwango cha wastani cha abiria. Bado kuna nafasi za kutosha.'
                }
            elif prediction < 65:
                return {
                    'level': 'wastani juu',
                    'description': 'Abiria wengi kidogo',
                    'advice': 'Basi linaweza kuwa na abiria wengi lakini bado linaweza kubebeka.'
                }
            else:
                return {
                    'level': 'msongamano',
                    'description': 'Msongamano mkuu',
                    'advice': 'Basi litakuwa limejaa kabisa. Fikiria kusafiri wakati mwingine au subiri basi jingine.'
                }
        else:
            if prediction < 20:
                return {
                    'level': 'very low',
                    'description': 'Nearly empty bus',
                    'advice': 'Perfect time to travel. You\'ll have plenty of space and a comfortable ride.'
                }
            elif prediction < 35:
                return {
                    'level': 'low',
                    'description': 'Few passengers',
                    'advice': 'Great travel conditions with adequate seating and fresh air.'
                }
            elif prediction < 50:
                return {
                    'level': 'moderate',
                    'description': 'Average passenger count',
                    'advice': 'Normal passenger levels with decent space availability.'
                }
            elif prediction < 65:
                return {
                    'level': 'moderately high',
                    'description': 'Getting crowded',
                    'advice': 'The bus will be fairly busy, but still manageable.'
                }
            else:
                return {
                    'level': 'high',
                    'description': 'Heavy crowding expected',
                    'advice': 'Expect a packed bus. Consider traveling at a different time or waiting for the next one.'
                }
    
    @staticmethod
    def generate_prediction_response(prediction, structured_data, language='English'):
        """Generate a comprehensive, conversational prediction response"""
        insights = ConversationalResponseGenerator.get_crowding_insights(prediction, structured_data, language)
        
        # Parse time for better display
        time_parts = structured_data['time'].split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        
        # Format time for display
        if language == 'Swahili':
            if hour == 0:
                time_display = f"usiku wa manane ({structured_data['time']})"
            elif hour < 12:
                time_display = f"saa {hour}:{minute:02d} asubuhi"
            elif hour == 12:
                time_display = f"saa 12:{minute:02d} mchana"
            else:
                time_display = f"saa {hour-12}:{minute:02d} jioni"
            
            day_display = {
                'Monday': 'Jumatatu', 'Tuesday': 'Jumanne', 'Wednesday': 'Jumatano',
                'Thursday': 'Alhamisi', 'Friday': 'Ijumaa', 'Saturday': 'Jumamosi', 'Sunday': 'Jumapili'
            }.get(structured_data['day'], structured_data['day'])
            
            response = f"🚌 Utabiri wa Abiria - {day_display}\n\n"
            response += f"⏰ Wakati: {time_display}\n"
            response += f"🌤 Hali ya hewa: {structured_data['weather']}\n"
            response += f"👥 Idadi inayotabiriwa: {prediction} abiria\n"
            response += f"📊 Kiwango cha msongamano: {insights['level'].title()}\n\n"
            response += f"💡 Uchambuzi wangu: {insights['advice']}\n\n"
            
            if structured_data['peak_hours'] == 'Yes':
                response += "⚠️ Kumbuka: Huu ni wakati wa msongamano mkuu, kwa hiyo tarajia abiria wengi zaidi.\n\n"
            
            response += "Je, unahitaji utabiri wa wakati mwingine au una swali lingine? 😊"
            
        else:
            if hour == 0:
                time_display = f"midnight ({structured_data['time']})"
            elif hour < 12:
                am_pm_hour = hour if hour != 0 else 12
                time_display = f"{am_pm_hour}:{minute:02d} AM"
            elif hour == 12:
                time_display = f"12:{minute:02d} PM"
            else:
                time_display = f"{hour-12}:{minute:02d} PM"
            
            response = f"🚌 Passenger Prediction for {structured_data['day']}\n\n"
            response += f"⏰ Time: {time_display}\n"
            response += f"🌤 Weather conditions: {structured_data['weather']}\n"
            response += f"👥 Predicted passengers: {prediction}\n"
            response += f"📊 Crowding level: {insights['level'].title()}\n\n"
            response += f"💡 My analysis: {insights['advice']}\n\n"
            
            if structured_data['peak_hours'] == 'Yes':
                response += "⚠️ Note: This is during peak hours, so expect higher passenger volumes.\n\n"
            
            response += "Would you like a prediction for a different time, or do you have any other questions? 😊"
        
        return response
    
    @staticmethod
    def generate_greeting_response(language='English'):
        """Generate a warm, engaging greeting"""
        greetings = {
            'English': [
                "👋 Hello! I'm your ChatBus AI Assistant, and I'm here to help you navigate bus travel with smart predictions.\n\nI specialize in predicting passenger flow patterns, so you can plan your journeys efficiently. Whether you're trying to avoid crowds or find the best travel times, I've got the insights you need.\n\n✨ What I can help you with:\n• Passenger count predictions for any day and time\n• Peak hour analysis and crowd level insights\n• Best travel time recommendations\n• Real-time travel advice\n\n💡 Try asking me something like:\n• \"How crowded will it be on Monday at 8 AM?\"\n• \"What's the best time to travel on Friday?\"\n• \"Will there be many passengers tomorrow evening?\"\n\nWhat would you like to know about your next bus journey? 😊",
                
                "🌟 Welcome to ChatBus AI! I'm your personal travel companion, here to make your bus journeys smoother and more predictable.\n\nThink of me as your travel planning assistant. I can predict passenger flows, identify the best travel times, and help you avoid those uncomfortable crowded rides.\n\n🚀 Ready to explore?\n• Ask about any specific day and time\n• Get insights on peak hours and quiet periods\n• Discover optimal travel windows\n• Learn about passenger patterns\n\n💬 Just tell me: When and where do you want to travel? I'll give you the complete analysis of what to expect. 😊"
            ],
            'Swahili': [
                "👋 Habari! Mimi ni ChatBus AI Assistant wako, na nipo hapa kukusaidia katika usafiri wa mabasi kwa kutumia utabiri wa akili.\n\nNina utaalamu wa kutabiri mifumo ya mtiririko wa abiria, ili uweze kupanga safari zako kwa ufanisi. Iwe unataka kuepuka msongamano au kutafuta nyakati bora za kusafiri, nina maarifa unayohitaji.\n\n✨ Ninachoweza kukusaidia:\n• Utabiri wa idadi ya abiria kwa siku na wakati wowote\n• Uchambuzi wa nyakati za msongamano na uelewa wa kiwango cha msongamano\n• Mapendekezo ya nyakati bora za kusafiri\n• Ushauri wa kusafiri wa wakati halisi\n\n💡 Jaribu kuniuliza kitu kama:\n• \"Kutakuwa na msongamano kiasi gani Jumatatu saa 8 asubuhi?\"\n• \"Ni wakati gani bora wa kusafiri Ijumaa?\"\n• \"Je, kutakuwa na abiria wengi kesho jioni?\"\n\nUngependa kujua nini kuhusu safari yako ijayo ya basi? 😊",
                
                "🌟 Karibu katika ChatBus AI! Mimi ni mwenza wako wa kibinafsi wa kusafiri, nipo hapa kufanya safari zako za mabasi ziwe laini na za kutabiriwa.\n\nNifikirike kama msaidizi wako wa kupanga safari. Ninaweza kutabiri mtiririko wa abiria, kutambua nyakati bora za kusafiri, na kukusaidia kuepuka safari za msongamano zisizotamanisha.\n\n🚀 Uko tayari kuchunguza?\n• Uliza kuhusu siku na wakati wowote mahususi\n• Pata maarifa kuhusu nyakati za msongamano na vipindi vya kimya\n• Gundua madirisha ya kusafiri bora\n• Jifunze kuhusu mifumo ya abiria\n\n💬 Niambie tu: Unataka kusafiri lini na wapi? Nitakupa uchambuzi kamili wa kile unachoweza kutarajia. 😊"
            ]
        }
        
        return random.choice(greetings[language])
    
    @staticmethod
    def generate_thank_you_response(language='English'):
        """Generate a warm thank you response"""
        responses = {
            'English': [
                "🙏 You're absolutely welcome! I'm so glad I could help make your travel planning easier.\n\nRemember, I'm always here whenever you need passenger predictions or travel insights. Whether it's for tomorrow's commute or planning a special trip, just give me a shout!\n\n🚌 Safe travels, and I hope your journey is comfortable and pleasant! 😊",
                
                "😊 My pleasure! It makes me happy to help fellow travelers make smarter journey decisions.\n\nDon't hesitate to come back anytime you need help with:\n• Planning your daily commute\n• Avoiding rush hour crowds\n• Finding the perfect travel windows\n• Any other bus-related questions!\n\n🌟 Wishing you smooth and comfortable travels ahead!"
            ],
            'Swahili': [
                "🙏 Karibu sana kabisa! Nimefurahi sana kwamba niliweza kusaidia kufanya mipango yako ya kusafiri iwe rahisi.\n\nKumbuka, nipo hapa kila wakati unavyohitaji utabiri wa abiria au maarifa ya kusafiri. Iwe ni kwa ajili ya safari za kesho au kupanga safari maalumu, niite tu!\n\n🚌 Safiri salama, na natumai safari yako itakuwa ya starehe na ya kupendeza! 😊",
                
                "😊 Furaha yangu! Inanifurahisha kusaidia wasafiri wenzangu kufanya maamuzi mazuri ya safari.\n\nUsisite kurudi wakati wowote unahitaji msaada na:\n• Kupanga safari zako za kila siku\n• Kuepuka msongamano wa nyakati za msongamano\n• Kutafuta madirisha kamili ya kusafiri\n• Maswali mengine yanayohusiana na mabasi!\n\n🌟 Nakutakia safari laini na za starehe mbele!"
            ]
        }
        
        return random.choice(responses[language])
    
    @staticmethod
    def generate_fallback_response(prompt, language='English'):
        """Generate helpful fallback response when the model doesn't understand"""
        if language == 'Swahili':
            response = "🤔 Pole, sielewi vizuri ulichomaanisha, lakini nina hamu ya kukusaidia!\n\n"
            response += "Mimi ni mtaalamu wa kutabiri idadi ya abiria katika mabasi. Ninaweza kukusaidia kwa:\n\n"
            response += "📋 Maswali yanayofaa:\n"
            response += "• \"Kuna abiria wangapi Jumatatu saa 8 asubuhi?\"\n"
            response += "• \"Je, ni wakati gani bora wa kusafiri Ijumaa?\"\n"
            response += "• \"Kutakuwa na msongamano kiasi gani kesho jioni?\"\n"
            response += "• \"Nitafute wakati wa kusafiri usio na msongamano Jumanne.\"\n\n"
            response += "💡 Miwongozo ya haraka:\n"
            response += "• Taja siku (Jumatatu, Jumanne, nk.)\n"
            response += "• Ongeza wakati (saa 8 asubuhi, jioni, nk.)\n"
            response += "• Unaweza pia kunitaja hali ya hewa ikiwa ni muhimu\n\n"
            response += "Jaribu tena kwa kutumia mfano wa hapo juu, au niambie tu unataka kujua nini kuhusu usafiri wa basi! 😊"
        else:
            response = "🤔 I'm not quite sure what you're asking, but I'm eager to help!\n\n"
            response += "I specialize in predicting passenger counts for bus travel. I can assist you with:\n\n"
            response += "📋 Great questions to ask:\n"
            response += "• \"How many passengers on Monday at 8 AM?\"\n"
            response += "• \"What's the best time to travel on Friday?\"\n"
            response += "• \"How crowded will it be tomorrow evening?\"\n"
            response += "• \"Find me a less crowded time to travel on Tuesday.\"\n\n"
            response += "💡 Quick tips:\n"
            response += "• Mention the day (Monday, Tuesday, etc.)\n"
            response += "• Add the time (8 AM, evening, etc.)\n"
            response += "• You can also mention weather conditions if relevant\n\n"
            response += "Try again using one of the examples above, or just tell me what you'd like to know about bus travel! 😊"
        
        return response

# API Endpoints
@app.route('/predict-from-prompt', methods=['POST'])
def predict_from_prompt():
    """
    Enhanced endpoint: Natural Language → Analysis → Response
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Empty prompt provided'}), 400
        
        prompt_lower = prompt.lower()
        language = NLPProcessor.detect_language(prompt)
        
        # Enhanced greeting detection
        greetings = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'how do you do', 'what\'s up', 'howdy', 'greetings',
            'introduce yourself', 'who are you', 'what are you', 'what can you do',
            'habari', 'mambo', 'hujambo', 'salamu', 'shikamoo', 'vipi', 'sasa'
        ]
        
        if any(greeting in prompt_lower for greeting in greetings):
            response_text = ConversationalResponseGenerator.generate_greeting_response(language)
            
            return jsonify({
                'predicted_passengers': 0,
                'response': response_text,
                'extracted_data': {},
                'language_detected': language,
                'message_type': 'greeting'
            })
        
        # Enhanced thank you detection
        thank_you_phrases = [
            'thank you', 'thanks', 'thank u', 'thanku', 'thx', 'ty',
            'appreciate', 'grateful', 'nice', 'good job', 'well done',
            'awesome', 'great', 'perfect', 'excellent', 'amazing',
            'asante', 'shukran', 'gracias', 'merci', 'asanteni'
        ]
        
        if any(phrase in prompt_lower for phrase in thank_you_phrases):
            response_text = ConversationalResponseGenerator.generate_thank_you_response(language)
            
            return jsonify({
                'predicted_passengers': 0,
                'response': response_text,
                'extracted_data': {},
                'language_detected': language,
                'message_type': 'thank_you'
            })
        
        # Check if this is a prediction request
        if NLPProcessor.is_prediction_request(prompt):
            # Step 1: Extract structured data from natural language
            structured_data = NLPProcessor.extract_structured_data(prompt)
            
            # Step 2: Get prediction from XGBoost model
            prediction = XGBoostPredictor.predict_passengers(structured_data)
            
            # Step 3: Generate conversational response
            response_text = ConversationalResponseGenerator.generate_prediction_response(
                prediction, structured_data, language
            )
            
            return jsonify({
                'predicted_passengers': prediction,
                'response': response_text,
                'extracted_data': structured_data,
                'language_detected': language,
                'message_type': 'prediction'
            })
        
        else:
            # Generate helpful fallback response
            response_text = ConversationalResponseGenerator.generate_fallback_response(prompt, language)
            
            return jsonify({
                'predicted_passengers': 0,
                'response': response_text,
                'extracted_data': {},
                'language_detected': language,
                'message_type': 'fallback'
            })
    
    except Exception as e:
        language = NLPProcessor.detect_language(prompt) if 'prompt' in locals() else 'English'
        
        if language == 'Swahili':
            error_response = "🔧 Samahani, nimepata tatizo la kiufundi. Tafadhali jaribu tena baada ya muda mfupi.\n\nIkiwa tatizo linaendelea, hakikisha kwamba server ya Flask inafanya kazi vizuri."
        else:
            error_response = "🔧 Sorry, I encountered a technical issue. Please try again in a moment.\n\nIf the problem persists, please ensure the Flask server is running properly."
        
        return jsonify({
            'predicted_passengers': 0,
            'response': error_response,
            'extracted_data': {},
            'language_detected': language,
            'message_type': 'error',
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/predict-structured', methods=['POST'])
def predict_structured():
    """
    Direct endpoint: Structured Data → XGBoost → Response
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No structured data provided'}), 400
        
        # Validate required fields
        required_fields = ['date', 'time', 'day', 'weather', 'peak_hours', 'weekends', 'holidays']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get prediction from XGBoost model
        prediction = XGBoostPredictor.predict_passengers(data)
        
        # Generate simple response
        if prediction < 30:
            level = 'low'
        elif prediction < 60:
            level = 'moderate'
        else:
            level = 'high'
        
        return jsonify({
            'predicted_passengers': prediction,
            'crowding_level': level,
            'input_data': data
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/extract-data', methods=['POST'])
def extract_data_only():
    """
    Utility endpoint: Natural Language → Structured Data (no prediction)
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Empty prompt provided'}), 400
        
        # Extract structured data from natural language
        structured_data = NLPProcessor.extract_structured_data(prompt)
        language = NLPProcessor.detect_language(prompt)
        
        return jsonify({
            'extracted_data': structured_data,
            'language_detected': language,
            'original_prompt': prompt,
            'is_prediction_request': NLPProcessor.is_prediction_request(prompt)
        })
    
    except Exception as e:
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'Enhanced ChatBus AI service with conversational responses is running',
        'features': ['ChatGPT-like Responses', 'Intelligent Fallbacks', 'Bilingual Support', 'NLP Processing', 'XGBoost Prediction'],
        'languages': ['English', 'Swahili'],
        'version': '2.0'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)