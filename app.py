import pickle
import numpy as np
import pandas as pd
import requests
import mysql.connector
from joblib import load
from flask import Flask, request, jsonify, render_template
from markupsafe import escape  
import json

app = Flask(__name__)

# Load the model and transformation files
potato_model = pickle.load(open("potato_model.pkl", "rb"))
columns = pickle.load(open("crop_columns.pkl", "rb"))
transformer = load(filename="potato_transformer.joblib")

# MySQL Database Connection
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "irrigation_db"
}

def connect_db():
    return mysql.connector.connect(**DB_CONFIG)

# OpenWeather API Config
OPENWEATHER_API_KEY = "87e0abd62d32f7d24c9be191a8850480"



def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(url)
        response_json = response.json()

        print("DEBUG: OpenWeather API Full Response:", json.dumps(response_json, indent=2))  # Debugging API response

        # Check if API request failed
        if "cod" in response_json and response_json["cod"] != "200":
            print(f"ERROR: OpenWeather API returned an error: {response_json.get('message', 'Unknown error')}")
            return None, None, "Weather data not available", None

        # Extract the first forecast entry
        forecast_list = response_json.get("list", [])
        if not forecast_list:
            return None, None, "No forecast data", None

        first_forecast = forecast_list[0]
        temp = first_forecast["main"]["temp"]
        humidity = first_forecast["main"]["humidity"]
        weather_desc = first_forecast["weather"][0]["description"]

        # Extract the next 7 time points (instead of full days)
        forecast = []
        for i, item in enumerate(forecast_list[:7]):  # Get 7 time points
            day_temp = item["main"]["temp"]
            day_humidity = item["main"]["humidity"]
            weather_desc = item["weather"][0]["description"]

            forecast.append({
                "day": f"Forecast {i+1}",
                "temp": day_temp,
                "humidity": day_humidity,
                "weather": weather_desc
            })

        return temp, humidity, weather_desc, forecast

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch weather data: {str(e)}")
        return None, None, "Weather API error", None

    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&exclude=minutely,hourly"
    response = requests.get(url).json()  # Fetch API response

    print("DEBUG: OpenWeather API Response:", response)  # Debugging

    # Check if the response contains forecast data
    if "daily" not in response:
        return None, None, "Weather data not available", None

    current_temp = response["current"]["temp"]
    current_humidity = response["current"]["humidity"]
    current_weather = response["current"]["weather"][0]["description"]

    # Extract the 7-day forecast
    forecast = []
    for day in response["daily"]:
        day_temp = day["temp"]["day"]
        day_humidity = day["humidity"]
        weather_desc = day["weather"][0]["description"]
        forecast.append({"temp": day_temp, "humidity": day_humidity, "weather": weather_desc})

    return current_temp, current_humidity, current_weather, forecast

    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()  # Fetch API response

    print("DEBUG: OpenWeather API Response:", response)  # Print API response for debugging

    # Check if "main" key exists in response
    if "main" not in response:
        return None, None, "Weather data not available"

    return response["main"]["temp"], response["main"]["humidity"], response["weather"][0]["description"]

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    lat = data.get("latitude", "").strip()
    lon = data.get("longitude", "").strip()
    crop_type = data.get("crop_type", "").strip()

    # Validate latitude and longitude
    if not lat or not lon:
        return "Error: Latitude and Longitude are required!", 400

    # Convert latitude and longitude to float
    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return "Error: Invalid Latitude or Longitude!", 400

    # Fetch weather data (current + forecast)
    temperature, humidity, weather_desc, forecast = get_weather(lat, lon)

    # Ensure weather data is valid
    if temperature is None or humidity is None:
        return "Error: Could not retrieve weather data!", 400

    print("DEBUG: Temperature:", temperature)
    print("DEBUG: Humidity:", humidity)
    print("DEBUG: Expected Columns:", columns)

    # Ensure categorical values are received properly
    soil_type = data.get("SOIL_TYPE", "").upper().strip()
    region = data.get("REGION", "").upper().strip()
    weather_condition = data.get("WEATHER_CONDITION", "").upper().strip()

    # Validate categorical values
    if not soil_type or not region or not weather_condition:
        return "Error: Missing categorical input values!", 400  # Return HTTP 400 Bad Request

    print("DEBUG: Raw Input Values ->", soil_type, region, weather_condition)

    # **Ensure categorical values remain in string format**
    soil_mapping = {"DRY": "DRY", "HUMID": "HUMID", "WET": "WET"}
    region_mapping = {"DESERT": "DESERT", "SEMI ARID": "SEMI ARID", "SEMI HUMID": "SEMI HUMID", "HUMID": "HUMID"}
    weather_mapping = {"NORMAL": "NORMAL", "SUNNY": "SUNNY", "WINDY": "WINDY", "RAINY": "RAINY"}

    # Convert to expected format
    soil_type = soil_mapping.get(soil_type, None)
    region = region_mapping.get(region, None)
    weather_condition = weather_mapping.get(weather_condition, None)

    # If any value is None, return an error
    if soil_type is None or region is None or weather_condition is None:
        return f"Error: Invalid categorical values! Received: {soil_type}, {region}, {weather_condition}", 400

    print(f"Converted Values -> SOIL_TYPE: {soil_type}, REGION: {region}, WEATHER_CONDITION: {weather_condition}")

    # Ensure TEMP_MIN and TEMP_MAX are provided and valid
    temp_min = data.get("TEMP_MIN", "").strip()
    temp_max = data.get("TEMP_MAX", "").strip()

    try:
        temp_min = float(temp_min) if temp_min else temperature  # Default to fetched temperature if missing
        temp_max = float(temp_max) if temp_max else temperature  # Default to fetched temperature if missing
    except ValueError:
        return "Error: Invalid temperature values!", 400

    # **Ensure DataFrame Columns Match Model Training Order**
    input_data = [[soil_type, region, weather_condition, temp_min, temp_max]]

    df = pd.DataFrame(input_data, columns=columns)
    print("DEBUG: Input DataFrame BEFORE Transformation:\n", df)
    print("DEBUG: Input DataFrame Data Types:\n", df.dtypes)

    # Ensure correct types
    df.fillna("", inplace=True)  # Avoid NaN errors
    df = df.astype(str)  # Ensure categorical values remain strings

    try:
        transformed_df = transformer.transform(df)  # **Fixed: Ensure correct format**
        print("DEBUG: Transformed DataFrame AFTER Transformation:\n", transformed_df)

        output = potato_model.predict(transformed_df)[0]
    except Exception as e:
        print("DEBUG: Transformation Error:", str(e))
        return f"Error: {str(e)}", 400

    return render_template("home.html", 
                           prediction_text=f"Water Required: {round(output, 2)} liters",
                           weather=f"Current Weather: {weather_desc}",
                           forecast=forecast)  # **Added Forecast Data**

    data = request.form
    lat, lon = data.get("latitude"), data.get("longitude")
    crop_type = data.get("crop_type")

    # Fetch real-time weather data + 7-day forecast
    temperature, humidity, weather_desc, forecast = get_weather(lat, lon)

    print("DEBUG: Temperature:", temperature)
    print("DEBUG: Humidity:", humidity)
    print("DEBUG: Expected Columns:", columns)

    # Ensure categorical values are received properly
    soil_type = data.get("SOIL_TYPE", "").upper().strip()
    region = data.get("REGION", "").upper().strip()
    weather_condition = data.get("WEATHER_CONDITION", "").upper().strip()

    if not soil_type or not region or not weather_condition:
        return "Error: Missing categorical input values!", 400  # Return HTTP 400 Bad Request

    print("DEBUG: Raw Input Values ->", soil_type, region, weather_condition)

    # **Ensure categorical values remain in string format**
    soil_mapping = {"DRY": "DRY", "HUMID": "HUMID", "WET": "WET"}
    region_mapping = {"DESERT": "DESERT", "SEMI ARID": "SEMI ARID", "SEMI HUMID": "SEMI HUMID", "HUMID": "HUMID"}
    weather_mapping = {"NORMAL": "NORMAL", "SUNNY": "SUNNY", "WINDY": "WINDY", "RAINY": "RAINY"}

    # Convert to expected format
    soil_type = soil_mapping.get(soil_type, soil_type)
    region = region_mapping.get(region, region)
    weather_condition = weather_mapping.get(weather_condition, weather_condition)

    print(f"Converted Values -> SOIL_TYPE: {soil_type}, REGION: {region}, WEATHER_CONDITION: {weather_condition}")

    # **Ensure DataFrame Columns Match Model Training Order**
    input_data = [[soil_type, region, weather_condition, temperature, humidity]]

    df = pd.DataFrame(input_data, columns=columns)
    print("DEBUG: Input DataFrame BEFORE Transformation:\n", df)
    print("DEBUG: Input DataFrame Data Types:\n", df.dtypes)

    # Ensure correct types
    df.fillna("", inplace=True)  # Avoid NaN errors
    df = df.astype(str)  # Ensure categorical values remain strings

    try:
        transformed_df = transformer.transform(df)  # **Fixed: Ensure correct format**
        print("DEBUG: Transformed DataFrame AFTER Transformation:\n", transformed_df)

        output = potato_model.predict(transformed_df)[0]
    except Exception as e:
        print("DEBUG: Transformation Error:", str(e))
        return f"Error: {str(e)}", 400

    return render_template("home.html", 
                           prediction_text=f"Water Required: {round(output, 2)} liters",
                           weather=f"Current Weather: {weather_desc}",
                           forecast=forecast)  # **Added Forecast Data**

    data = request.form
    lat, lon = data.get("latitude"), data.get("longitude")
    crop_type = data.get("crop_type")

    # Fetch real-time weather data
    temperature, humidity, weather_desc = get_weather(lat, lon)

    print("DEBUG: Temperature:", temperature)
    print("DEBUG: Humidity:", humidity)
    print("DEBUG: Expected Columns:", columns)

    # Ensure categorical values are received properly
    soil_type = data.get("SOIL_TYPE", "").upper().strip()
    region = data.get("REGION", "").upper().strip()
    weather_condition = data.get("WEATHER_CONDITION", "").upper().strip()

    if not soil_type or not region or not weather_condition:
        return "Error: Missing categorical input values!", 400  # Return HTTP 400 Bad Request

    print("DEBUG: Raw Input Values ->", soil_type, region, weather_condition)

    # **Convert Categorical Encodings Back to Original Strings**
    soil_mapping = {0: "DRY", 1: "HUMID", 2: "WET"}
    region_mapping = {0: "DESERT", 1: "SEMI ARID", 2: "SEMI HUMID", 3: "HUMID"}
    weather_mapping = {0: "NORMAL", 1: "SUNNY", 2: "WINDY", 3: "RAINY"}

    # Ensure values remain in original string format
    soil_type = soil_mapping.get(soil_type, soil_type)
    region = region_mapping.get(region, region)
    weather_condition = weather_mapping.get(weather_condition, weather_condition)

    print(f"Converted Values -> SOIL_TYPE: {soil_type}, REGION: {region}, WEATHER_CONDITION: {weather_condition}")

    # **Ensure DataFrame Columns Match Model Training Order**
    input_data = [[soil_type, region, weather_condition, temperature, humidity]]

    df = pd.DataFrame(input_data, columns=columns)
    print("DEBUG: Input DataFrame BEFORE Transformation:\n", df)
    print("DEBUG: Input DataFrame Data Types:\n", df.dtypes)

    # **Ensure Correct Data Format**
    df.fillna("", inplace=True)  # Prevent NaN errors
    df = df.astype(str)  # Convert categorical values to strings before transformation

    try:
        transformed_df = transformer.transform(df)  # **Fixed: Ensure correct format**
        print("DEBUG: Transformed DataFrame AFTER Transformation:\n", transformed_df)

        output = potato_model.predict(transformed_df)[0]
    except Exception as e:
        print("DEBUG: Transformation Error:", str(e))
        return f"Error: {str(e)}", 400

    return render_template("home.html", prediction_text=f"Water Required: {round(output, 2)} liters", weather=f"Weather: {weather_desc}")

    data = request.form
    lat, lon = data.get("latitude"), data.get("longitude")
    crop_type = data.get("crop_type")

    # Fetch real-time weather data
    temperature, humidity, weather_desc = get_weather(lat, lon)

    print("DEBUG: Temperature:", temperature)
    print("DEBUG: Humidity:", humidity)
    print("DEBUG: Expected Columns:", columns)

    # Ensure input values are not None
    soil_type = data.get("SOIL_TYPE")
    region = data.get("REGION")
    weather_condition = data.get("WEATHER_CONDITION")

    if not soil_type or not region or not weather_condition:
        return "Error: Missing categorical input values!", 400  # Return HTTP 400 Bad Request

    # Convert categorical values to uppercase and remove spaces
    soil_type = soil_type.upper().strip()
    region = region.upper().strip()
    weather_condition = weather_condition.upper().strip()

    # Encode categorical variables into numerical values
    soil_mapping = {"DRY": 0, "HUMID": 1, "WET": 2}
    region_mapping = {"DESERT": 0, "SEMI ARID": 1, "SEMI HUMID": 2, "HUMID": 3}
    weather_mapping = {"NORMAL": 0, "SUNNY": 1, "WINDY": 2, "RAINY": 3}

    if soil_type not in soil_mapping or region not in region_mapping or weather_condition not in weather_mapping:
        return f"Error: Invalid categorical values! Received: {soil_type}, {region}, {weather_condition}", 400

    soil_type = soil_mapping[soil_type]
    region = region_mapping[region]
    weather_condition = weather_mapping[weather_condition]

    print(f"Encoded Values -> SOIL_TYPE: {soil_type}, REGION: {region}, WEATHER_CONDITION: {weather_condition}")

    # Create input DataFrame with correct column order and values
    input_data = {
        "SOIL TYPE": [soil_type],
        "REGION": [region],
        "WEATHER CONDITION": [weather_condition],
        "TEMP_MIN": [temperature],
        "TEMP_MAX": [temperature]  # Assuming TEMP_MIN and TEMP_MAX are the same for simplicity
    }

    df = pd.DataFrame(input_data, columns=columns)
    print("DEBUG: Input DataFrame BEFORE Transformation:\n", df)
    print("DEBUG: Input DataFrame Data Types:\n", df.dtypes)

    # Debug transformed data
    try:
        transformed_df = transformer.transform(df)
        print("DEBUG: Transformed DataFrame AFTER Transformation:\n", transformed_df)
    except Exception as e:
        print("DEBUG: Transformation Error:", e)
        return f"Transformation Error: {str(e)}", 400

    output = potato_model.predict(transformed_df)[0]

    return render_template("home.html", prediction_text=f"Water Required: {round(output, 2)} liters", weather=f"Weather: {weather_desc}")
    data = request.form
    lat, lon = data.get("latitude"), data.get("longitude")
    crop_type = data.get("crop_type")

    # Fetch real-time weather data
    temperature, humidity, weather_desc = get_weather(lat, lon)

    print("DEBUG: Temperature:", temperature)
    print("DEBUG: Humidity:", humidity)
    print("DEBUG: Expected Columns:", columns)

    # Ensure input values are not None
    soil_type = data.get("SOIL_TYPE")
    region = data.get("REGION")
    weather_condition = data.get("WEATHER_CONDITION")

    if not soil_type or not region or not weather_condition:
        return "Error: Missing categorical input values!", 400  # Return HTTP 400 Bad Request

    # Convert categorical values to uppercase and remove spaces
    soil_type = soil_type.upper().strip()
    region = region.upper().strip()
    weather_condition = weather_condition.upper().strip()

    # Encode categorical variables into numerical values
    soil_mapping = {"DRY": 0, "HUMID": 1, "WET": 2}
    region_mapping = {"DESERT": 0, "SEMI ARID": 1, "SEMI HUMID": 2, "HUMID": 3}
    weather_mapping = {"NORMAL": 0, "SUNNY": 1, "WINDY": 2, "RAINY": 3}

    if soil_type not in soil_mapping or region not in region_mapping or weather_condition not in weather_mapping:
        return f"Error: Invalid categorical values! Received: {soil_type}, {region}, {weather_condition}", 400

    soil_type = soil_mapping[soil_type]
    region = region_mapping[region]
    weather_condition = weather_mapping[weather_condition]

    print(f"Encoded Values -> SOIL_TYPE: {soil_type}, REGION: {region}, WEATHER_CONDITION: {weather_condition}")

    # Create input DataFrame with encoded values
    input_data = [[temperature, humidity, soil_type, region, weather_condition]]

    df = pd.DataFrame(input_data, columns=columns)
    print("DEBUG: Input DataFrame BEFORE Transformation:\n", df)
    print("DEBUG: Input DataFrame Data Types:\n", df.dtypes)

    # Ensure all values are numerical before transformation
    df = df.astype(float)

    # Debug transformed data
    try:
        transformed_df = transformer.transform(df)
        print("DEBUG: Transformed DataFrame AFTER Transformation:\n", transformed_df)
    except Exception as e:
        print("DEBUG: Transformation Error:", e)
        return f"Transformation Error: {str(e)}", 400

    output = potato_model.predict(transformed_df)[0]

    return render_template("home.html", prediction_text=f"Water Required: {round(output, 2)} liters", weather=f"Weather: {weather_desc}")
    data = request.form
    lat, lon = data.get("latitude"), data.get("longitude")
    crop_type = data.get("crop_type")

    # Fetch real-time weather data
    temperature, humidity, weather_desc = get_weather(lat, lon)

    print("DEBUG: Temperature:", temperature)
    print("DEBUG: Humidity:", humidity)
    print("DEBUG: Expected Columns:", columns)

    # Ensure categorical values are received properly
    soil_type = data.get("SOIL_TYPE")
    region = data.get("REGION")
    weather_condition = data.get("WEATHER_CONDITION")

    if not soil_type or not region or not weather_condition:
        return "Error: Missing categorical input values!", 400  # Return HTTP 400 Bad Request

    # Convert categorical values to uppercase and remove spaces
    soil_type = soil_type.upper().strip()
    region = region.upper().strip()
    weather_condition = weather_condition.upper().strip()

    # Encode categorical variables into numerical values
    soil_mapping = {"DRY": 0, "HUMID": 1, "WET": 2}
    region_mapping = {"DESERT": 0, "SEMI ARID": 1, "SEMI HUMID": 2, "HUMID": 3}
    weather_mapping = {"NORMAL": 0, "SUNNY": 1, "WINDY": 2, "RAINY": 3}

    if soil_type not in soil_mapping or region not in region_mapping or weather_condition not in weather_mapping:
        return f"Error: Invalid categorical values! Received: {soil_type}, {region}, {weather_condition}", 400

    soil_type = soil_mapping[soil_type]
    region = region_mapping[region]
    weather_condition = weather_mapping[weather_condition]

    print(f"Encoded Values -> SOIL_TYPE: {soil_type}, REGION: {region}, WEATHER_CONDITION: {weather_condition}")

    # Create input DataFrame with encoded values
    input_data = [[temperature, humidity, soil_type, region, weather_condition]]

    df = pd.DataFrame(input_data, columns=columns)
    print("DEBUG: Input DataFrame BEFORE Transformation:\n", df)

    # Convert all values to float before transformation
    df = df.astype(float)

    transformed_df = transformer.transform(df)
    print("DEBUG: Transformed DataFrame AFTER Transformation:\n", transformed_df)

    output = potato_model.predict(transformed_df)[0]

    return render_template("home.html", prediction_text=f"Water Required: {round(output, 2)} liters", weather=f"Weather: {weather_desc}")

    data = request.form
    lat, lon = data.get("latitude"), data.get("longitude")
    crop_type = data.get("crop_type")

    # Fetch real-time weather data
    temperature, humidity, weather_desc = get_weather(lat, lon)

    print("DEBUG: Temperature:", temperature)
    print("DEBUG: Humidity:", humidity)
    print("DEBUG: Expected Columns:", columns)

    # Ensure input values are not None
    soil_type = data.get("SOIL_TYPE")
    region = data.get("REGION")
    weather_condition = data.get("WEATHER_CONDITION")

    if not soil_type or not region or not weather_condition:
        return "Error: Missing categorical input values!", 400  # Return HTTP 400 Bad Request

    # Convert categorical values to lowercase & check for unexpected inputs
    soil_type = soil_type.upper().strip()
    region = region.upper().strip()
    weather_condition = weather_condition.upper().strip()

    print("DEBUG: Raw Input Values ->", soil_type, region, weather_condition)

    # Convert categorical values to ensure compatibility with the trained model
    valid_soil_types = ["DRY", "HUMID", "WET"]
    valid_regions = ["DESERT", "SEMI ARID", "SEMI HUMID", "HUMID"]
    valid_weather_conditions = ["NORMAL", "SUNNY", "WINDY", "RAINY"]

    if soil_type not in valid_soil_types or region not in valid_regions or weather_condition not in valid_weather_conditions:
        return f"Error: Invalid categorical values! Received: {soil_type}, {region}, {weather_condition}", 400

    # Create input DataFrame with categorical values as strings (needed for OneHotEncoder)
    input_data = [[temperature, humidity, soil_type, region, weather_condition]]

    df = pd.DataFrame(input_data, columns=columns)
    print("DEBUG: Input DataFrame BEFORE Transformation:\n", df)

    # Ensure there are no NaN values before transformation
    df.fillna("", inplace=True)

    # Convert all values to string before transformation (if OneHotEncoder expects strings)
    df = df.astype(str)

    # Debug transformed data
    transformed_df = transformer.transform(df)
    print("DEBUG: Transformed DataFrame AFTER Transformation:\n", transformed_df)

    output = potato_model.predict(transformed_df)[0]

    return render_template("home.html", prediction_text=f"Water Required: {round(output, 2)} liters", weather=f"Weather: {weather_desc}")

    data = request.form
    lat, lon = data.get("latitude"), data.get("longitude")
    crop_type = data.get("crop_type")

    # Fetch real-time weather data
    temperature, humidity, weather_desc = get_weather(lat, lon)

    print("DEBUG: Temperature:", temperature)
    print("DEBUG: Humidity:", humidity)
    print("DEBUG: Expected Columns:", columns)

    # Ensure input values are not None
    soil_type = data.get("SOIL_TYPE")
    region = data.get("REGION")
    weather_condition = data.get("WEATHER_CONDITION")

    if not soil_type or not region or not weather_condition:
        return "Error: Missing categorical input values!", 400  # Return HTTP 400 Bad Request

    # Convert categorical values to lowercase & check for unexpected inputs
    soil_type = soil_type.upper().strip()
    region = region.upper().strip()
    weather_condition = weather_condition.upper().strip()

    soil_mapping = {"DRY": 0, "HUMID": 1, "WET": 2}
    region_mapping = {"DESERT": 0, "SEMI ARID": 1, "SEMI HUMID": 2, "HUMID": 3}
    weather_mapping = {"NORMAL": 0, "SUNNY": 1, "WINDY": 2, "RAINY": 3}

    # Convert categorical values using predefined mappings
    soil_type = soil_mapping.get(soil_type, None)
    region = region_mapping.get(region, None)
    weather_condition = weather_mapping.get(weather_condition, None)

    # Handle unknown values
    if soil_type is None or region is None or weather_condition is None:
        return "Error: Invalid categorical values provided!", 400

    print(f"Encoded Values -> SOIL_TYPE: {soil_type}, REGION: {region}, WEATHER_CONDITION: {weather_condition}")

    # Ensure input data matches expected features
    input_data = [[temperature, humidity, soil_type, region, weather_condition]]

    # Convert to DataFrame
    df = pd.DataFrame(input_data, columns=columns)
    print("DEBUG: Input DataFrame\n", df)

    # Ensure all values are numerical before transformation
    df = df.astype(float)

    transformed_df = transformer.transform(df)  # FIX: Ensure correct data format
    output = potato_model.predict(transformed_df)[0]

    return render_template("home.html", prediction_text=f"Water Required: {round(output, 2)} liters", weather=f"Weather: {weather_desc}")

    data = request.form
    lat, lon = data.get("latitude"), data.get("longitude")
    crop_type = data.get("crop_type")

    # Fetch real-time weather data
    temperature, humidity, weather_desc = get_weather(lat, lon)

    print("DEBUG: Temperature:", temperature)
    print("DEBUG: Humidity:", humidity)
    print("DEBUG: Expected Columns:", columns)

    # Convert categorical features into numerical encoding
    soil_mapping = {"DRY": 0, "HUMID": 1, "WET": 2}
    region_mapping = {"DESERT": 0, "SEMI ARID": 1, "SEMI HUMID": 2, "HUMID": 3}
    weather_mapping = {"NORMAL": 0, "SUNNY": 1, "WINDY": 2, "RAINY": 3}

    soil_type = soil_mapping.get(data.get("SOIL_TYPE"), 0)
    region = region_mapping.get(data.get("REGION"), 0)
    weather_condition = weather_mapping.get(data.get("WEATHER_CONDITION"), 0)

    print(f"Encoded Values -> SOIL_TYPE: {soil_type}, REGION: {region}, WEATHER_CONDITION: {weather_condition}")

    # Ensure input data matches expected features
    input_data = [[temperature, humidity, soil_type, region, weather_condition]]
    
    df = pd.DataFrame(input_data, columns=columns)
    print("DEBUG: Input DataFrame\n", df)

    transformed_df = transformer.transform(df)
    output = potato_model.predict(transformed_df)[0]

    return render_template("home.html", prediction_text=f"Water Required: {round(output, 2)} liters", weather=f"Weather: {weather_desc}")

    data = request.form
    lat, lon = data.get("latitude"), data.get("longitude")
    crop_type = data.get("crop_type")

    # Fetch real-time weather data
    temperature, humidity, weather_desc = get_weather(lat, lon)

    print("DEBUG: Temperature:", temperature)  # Debugging
    print("DEBUG: Humidity:", humidity)
    print("DEBUG: Expected Columns:", columns)

    # Ensure input data matches expected features
    input_data = [[temperature, humidity, data.get("SOIL_TYPE"), data.get("REGION"), data.get("WEATHER_CONDITION")]]
    
    df = pd.DataFrame(input_data, columns=columns)  # Ensure correct column names
    print("DEBUG: Input DataFrame\n", df)  # Debugging

    transformed_df = transformer.transform(df)
    output = potato_model.predict(transformed_df)[0]

    return render_template("home.html", prediction_text=f"Water Required: {round(output, 2)} liters", weather=f"Weather: {weather_desc}")

    data = request.form
    lat, lon = data.get("latitude"), data.get("longitude")
    crop_type = data.get("crop_type")
    
    # Fetch real-time weather data
    temperature, humidity, weather_desc = get_weather(lat, lon)
    
    # Prepare input data
    input_data = np.array([temperature, humidity]).reshape(1, -1)
    df = pd.DataFrame(input_data, columns=columns)
    transformed_df = transformer.transform(df)
    output = potato_model.predict(transformed_df)[0]
    
    # Store prediction in MySQL database
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (crop, water_required, latitude, longitude, weather_desc) VALUES (%s, %s, %s, %s, %s)",
                   (crop_type, output, lat, lon, weather_desc))
    conn.commit()
    conn.close()
    
    return render_template("home.html", prediction_text=f"Water Required: {round(output, 2)} liters", weather=f"Weather: {weather_desc}")

if __name__ == "__main__":
    app.run(debug=True)
