import os
import sys
import json
import random
from pathlib import Path
import pandas as pd
from flask import Flask, request, jsonify
from functools import wraps

# --- Configuration ---
FOOD_CSV_PATH = Path("foodData.csv")
RANDOM_SEED = 42
MAX_DISH_REPEAT_PER_WEEK = 2

# --- API Key Configuration ---
# Load valid API keys from an environment variable for better security.
# The variable should contain a comma-separated list of keys.
API_KEYS_STR = os.getenv("VALID_API_KEYS", "")
VALID_API_KEYS = {key.strip() for key in API_KEYS_STR.split(',') if key.strip()}

# --- Initialize Flask App ---
app = Flask(__name__)

# --- API Key Decorator ---
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key in VALID_API_KEYS:
            return f(*args, **kwargs)
        else:
            return jsonify({"error": "Unauthorized. A valid API key is required."}), 401
    return decorated_function

# --- Helper Functions ---
def safe_lower(text):
    return "" if pd.isna(text) else str(text).lower()

def load_and_clean_food_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Error: The food data file was not found at '{file_path}'")
    df = pd.read_csv(file_path, on_bad_lines='skip', skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]
    numeric_cols = ["Calories_kcal", "Protein_g", "Carbohydrates_g", "Fat_g", "Fiber_g"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["Dish_Name", "Category", "Ayurvedic_Rasa", "Ayurvedic_Virya", "Notes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def calculate_calorie_target(patient):
    try:
        weight_kg = float(patient["vitals"]["weight_kg"])
        height_cm = float(patient["vitals"]["height_cm"])
        age = int(patient["personalInfo"]["age"])
        gender = safe_lower(patient["personalInfo"]["gender"])
        activity_level = safe_lower(patient["lifestyle"]["physicalActivity"])
        activity_factors = {"low": 1.2, "moderate": 1.55, "high": 1.725}
        activity_multiplier = activity_factors.get(activity_level, 1.375)
        if gender.startswith("m"):
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
        maintenance_calories = bmr * activity_multiplier
        target_calories = maintenance_calories * 0.85
        return int(round(target_calories))
    except (KeyError, ValueError):
        return 2000

def apply_filters_to_food_df(df, patient):
    filtered_df = df.copy()
    dosha_imbalance = safe_lower(patient["ayurvedaProfile"]["doshaImbalance"])
    conditions = [safe_lower(c) for c in patient["medicalInfo"]["conditions"]]
    if "pitta" in dosha_imbalance:
        filtered_df = filtered_df[~filtered_df["Ayurvedic_Virya"].apply(lambda x: "hot" in safe_lower(x))]
        for rasa in ["pungent", "sour", "salty"]:
            filtered_df = filtered_df[~filtered_df["Ayurvedic_Rasa"].apply(lambda x: rasa in safe_lower(x))]
    if any(c in ["hypertension", "high blood pressure"] for c in conditions):
        filtered_df = filtered_df[~filtered_df["Ayurvedic_Rasa"].apply(lambda x: "salty" in safe_lower(x))]
    if any(c in ["diabetes", "prediabetes"] for c in conditions):
        filtered_df = filtered_df[~filtered_df["Ayurvedic_Guna"].apply(lambda x: "difficult" in safe_lower(x))]
        filtered_df = filtered_df[~filtered_df["Category"].apply(lambda x: "desserts" in safe_lower(x))]
    if "vegetarian" in safe_lower(patient["lifestyle"]["dietType"]):
        non_veg_keywords = ["chicken", "fish", "mutton", "egg", "prawn"]
        for keyword in non_veg_keywords:
            filtered_df = filtered_df[~filtered_df["Dish_Name"].apply(lambda x: keyword in safe_lower(x))]
    filtered_df = filtered_df[filtered_df["Calories_kcal"] > 10]
    if filtered_df.empty:
        raise ValueError("No food items remaining after applying filters. The criteria might be too strict.")
    return filtered_df

def generate_7_day_plan(filtered_df, patient):
    calorie_target = calculate_calorie_target(patient)
    meal_templates = {
        "breakfast": [["Breakfast", "Snacks"]],
        "lunch": [["Grains", "Breads"], ["Lentils (Dal)", "Main Course"], ["Vegetables (Sabzi)"], ["Beverages", "Condiments"]],
        "dinner": [["Soups", "Healing Meals"], ["Vegetables (Sabzi)"]]
    }
    plan = {
        "plan_metadata": { "patient_name": patient["personalInfo"]["fullName"], "plan_duration_days": 7, "primary_goal": patient["goals"]["shortTerm"], "daily_calorie_target_kcal": calorie_target },
        "weekly_plan": []
    }
    used_dishes_count = {}
    for day_num in range(1, 8):
        day_plan = {"day": day_num, "daily_total_calories_kcal": 0, "meals": {}}
        for meal_name, slots in meal_templates.items():
            meal_items = []
            for slot_categories in slots:
                candidates = filtered_df[filtered_df["Category"].isin(slot_categories)].copy()
                available_candidates = candidates[~candidates["Dish_ID"].isin([ dish_id for dish_id, count in used_dishes_count.items() if count >= MAX_DISH_REPEAT_PER_WEEK ])]
                if available_candidates.empty: available_candidates = candidates
                if not available_candidates.empty:
                    chosen_dish = available_candidates.sample(n=1, random_state=random.randint(1, 1000)).iloc[0]
                    dish_id = int(chosen_dish["Dish_ID"])
                    meal_items.append({ "Dish_ID": dish_id, "Dish_Name": chosen_dish["Dish_Name"], "Category": chosen_dish["Category"], "Calories_kcal": int(chosen_dish["Calories_kcal"]), "Notes": chosen_dish["Notes"].strip() })
                    used_dishes_count[dish_id] = used_dishes_count.get(dish_id, 0) + 1
            day_plan["meals"][meal_name] = {"items": meal_items}
        total_calories = sum(item["Calories_kcal"] for meal in day_plan["meals"].values() for item in meal["items"])
        day_plan["daily_total_calories_kcal"] = total_calories
        plan["weekly_plan"].append(day_plan)
    return plan


# --- Load Data and Perform Checks ONCE at Startup ---
try:
    food_df = load_and_clean_food_data(FOOD_CSV_PATH)
    print("INFO: Food database loaded successfully.")
    # Check if API keys are configured, otherwise the app is insecure
    if not VALID_API_KEYS:
        print("CRITICAL_ERROR: No VALID_API_KEYS found in environment variables.")
        print("Please set the VALID_API_KEYS environment variable to run the server securely.")
        sys.exit(1) # Exit if no keys are configured
    print("INFO: API keys loaded successfully.")
except FileNotFoundError as e:
    print(f"CRITICAL_ERROR: {e}")
    sys.exit(1) # Exit if the essential food data is missing


# --- API Endpoint Definition ---
@app.route('/generate_plan', methods=['POST'])
@require_api_key
def create_diet_plan():
    patient_data = request.get_json()
    if not patient_data:
        return jsonify({"error": "Invalid request. Please provide patient data in JSON format."}), 400

    try:
        suitable_food_df = apply_filters_to_food_df(food_df, patient_data)
        diet_plan = generate_7_day_plan(suitable_food_df, patient_data)
        return jsonify(diet_plan), 200
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Could not generate plan: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

# --- Main Execution Block for Local Development ---
if __name__ == '__main__':
    # This block is for running the app locally for testing.
    # A production deployment would use a WSGI server like Gunicorn instead.
    print("INFO: Starting Flask development server...")
    app.run(debug=True, port=5000)