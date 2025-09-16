import json
import random
from pathlib import Path
import pandas as pd

# --- Configuration ---
# Set file paths. Assumes files are in the same directory as the script.
FOOD_CSV_PATH = Path("foodData.csv")
PATIENT_JSON_PATH = Path("patientData.json")

# Constants for the diet plan generation
RANDOM_SEED = 42
MAX_DISH_REPEAT_PER_WEEK = 2 # How many times the same dish can appear in the week

# --- Helper Functions ---
def safe_lower(text):
    """Safely converts input to lowercase string, handling potential None or non-string types."""
    return "" if pd.isna(text) else str(text).lower()

def load_and_clean_food_data(file_path):
    """Loads the food CSV, cleans column names, and standardizes data types."""
    if not file_path.exists():
        raise FileNotFoundError(f"Error: The food data file was not found at '{file_path}'")

    df = pd.read_csv(file_path, on_bad_lines='skip', skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]

    # Coerce numeric columns to numbers, filling errors with 0
    numeric_cols = ["Calories_kcal", "Protein_g", "Carbohydrates_g", "Fat_g", "Fiber_g"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Normalize text fields for consistent matching
    for col in ["Dish_Name", "Category", "Ayurvedic_Rasa", "Ayurvedic_Virya", "Notes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    return df

def calculate_calorie_target(patient):
    """Calculates the daily calorie target using Mifflin-St Jeor BMR formula."""
    try:
        weight_kg = float(patient["vitals"]["weight_kg"])
        height_cm = float(patient["vitals"]["height_cm"])
        age = int(patient["personalInfo"]["age"])
        gender = safe_lower(patient["personalInfo"]["gender"])

        # Determine activity factor based on lifestyle
        activity_level = safe_lower(patient["lifestyle"]["physicalActivity"])
        activity_factors = {"low": 1.2, "moderate": 1.55, "high": 1.725}
        activity_multiplier = activity_factors.get(activity_level, 1.375) # Default to light activity

        # Mifflin-St Jeor Equation for BMR
        if gender.startswith("m"):
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

        # Adjust for goals: create a moderate deficit for weight management
        maintenance_calories = bmr * activity_multiplier
        # A 15% deficit is a common starting point for sustainable weight loss
        target_calories = maintenance_calories * 0.85 
        
        return int(round(target_calories))
    except (KeyError, ValueError) as e:
        print(f"Warning: Could not calculate calorie target due to missing/invalid data: {e}. Defaulting to 2000 kcal.")
        return 2000

def apply_filters_to_food_df(df, patient):
    """Filters the food dataframe based on the patient's Ayurvedic profile and medical conditions."""
    filtered_df = df.copy()
    
    # Extract patient info
    dosha_imbalance = safe_lower(patient["ayurvedaProfile"]["doshaImbalance"])
    conditions = [safe_lower(c) for c in patient["medicalInfo"]["conditions"]]
    
    # 1. Ayurvedic Dosha Filter
    if "pitta" in dosha_imbalance:
        # Avoid heating foods for Pitta imbalance
        filtered_df = filtered_df[~filtered_df["Ayurvedic_Virya"].apply(lambda x: "hot" in safe_lower(x))]
        # Avoid Pitta-aggravating tastes
        for rasa in ["pungent", "sour", "salty"]:
            filtered_df = filtered_df[~filtered_df["Ayurvedic_Rasa"].apply(lambda x: rasa in safe_lower(x))]

    # 2. Medical Condition Filter
    if any(c in ["hypertension", "high blood pressure"] for c in conditions):
        # Avoid salty taste and specific high-salt items
        filtered_df = filtered_df[~filtered_df["Ayurvedic_Rasa"].apply(lambda x: "salty" in safe_lower(x))]

    if any(c in ["diabetes", "prediabetes"] for c in conditions):
        # Avoid items noted as difficult to digest or overly sweet
        filtered_df = filtered_df[~filtered_df["Ayurvedic_Guna"].apply(lambda x: "difficult" in safe_lower(x))]
        filtered_df = filtered_df[~filtered_df["Category"].apply(lambda x: "desserts" in safe_lower(x))]
    
    # 3. Dietary Preference Filter
    if "vegetarian" in safe_lower(patient["lifestyle"]["dietType"]):
        # The provided data is already vegetarian, but this is good practice
        non_veg_keywords = ["chicken", "fish", "mutton", "egg", "prawn"]
        for keyword in non_veg_keywords:
            filtered_df = filtered_df[~filtered_df["Dish_Name"].apply(lambda x: keyword in safe_lower(x))]

    # 4. General Exclusions
    # Exclude items with zero calories as they can't be part of a meal plan
    filtered_df = filtered_df[filtered_df["Calories_kcal"] > 10]
    
    if filtered_df.empty:
        raise ValueError("No food items remaining after applying filters. The criteria might be too strict.")
        
    return filtered_df

def generate_7_day_plan(filtered_df, patient):
    """Main function to generate the full 7-day diet plan JSON object."""
    
    calorie_target = calculate_calorie_target(patient)
    
    # Meal structure template
    meal_templates = {
        "breakfast": [["Breakfast", "Snacks"]],
        "lunch": [["Grains", "Breads"], ["Lentils (Dal)", "Main Course"], ["Vegetables (Sabzi)"], ["Beverages", "Condiments"]],
        "dinner": [["Soups", "Healing Meals"], ["Vegetables (Sabzi)"]]
    }
    
    plan = {
        "plan_metadata": {
            "patient_name": patient["personalInfo"]["fullName"],
            "plan_duration_days": 7,
            "primary_goal": patient["goals"]["shortTerm"],
            "daily_calorie_target_kcal": calorie_target
        },
        "weekly_plan": []
    }
    
    used_dishes_count = {}

    for day_num in range(1, 8):
        day_plan = {"day": day_num, "daily_total_calories_kcal": 0, "meals": {}}
        
        for meal_name, slots in meal_templates.items():
            meal_items = []
            
            for slot_categories in slots:
                # Find candidate dishes for this slot
                candidates = filtered_df[filtered_df["Category"].isin(slot_categories)].copy()
                
                # Filter out dishes that have been used too often
                available_candidates = candidates[~candidates["Dish_ID"].isin([
                    dish_id for dish_id, count in used_dishes_count.items() if count >= MAX_DISH_REPEAT_PER_WEEK
                ])]
                
                # If filtering leaves no options, fall back to the original candidate pool for this slot
                if available_candidates.empty:
                    available_candidates = candidates
                
                if not available_candidates.empty:
                    # Randomly select one dish from the available pool
                    chosen_dish = available_candidates.sample(n=1, random_state=random.randint(1, 1000)).iloc[0]
                    
                    dish_id = int(chosen_dish["Dish_ID"])
                    meal_items.append({
                        "Dish_ID": dish_id,
                        "Dish_Name": chosen_dish["Dish_Name"],
                        "Category": chosen_dish["Category"],
                        "Calories_kcal": int(chosen_dish["Calories_kcal"]),
                        "Notes": chosen_dish["Notes"].strip()
                    })
                    
                    # Update the count for the chosen dish
                    used_dishes_count[dish_id] = used_dishes_count.get(dish_id, 0) + 1
            
            day_plan["meals"][meal_name] = {"items": meal_items}

        # Calculate total daily calories after planning all meals
        total_calories = sum(item["Calories_kcal"] for meal in day_plan["meals"].values() for item in meal["items"])
        day_plan["daily_total_calories_kcal"] = total_calories
        
        plan["weekly_plan"].append(day_plan)
        
    return plan

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Set a seed for reproducibility
        random.seed(RANDOM_SEED)

        # 1. Load Patient Data
        if not PATIENT_JSON_PATH.exists():
            raise FileNotFoundError(f"Error: Patient data file not found at '{PATIENT_JSON_PATH}'")
        with open(PATIENT_JSON_PATH, "r", encoding="utf-8") as f:
            patient_data = json.load(f)

        # 2. Load and Prepare Food Database
        food_df = load_and_clean_food_data(FOOD_CSV_PATH)
        
        # 3. Apply Filters to get suitable foods
        suitable_food_df = apply_filters_to_food_df(food_df, patient_data)
        
        # 4. Generate the 7-Day Diet Plan
        diet_plan = generate_7_day_plan(suitable_food_df, patient_data)
        
        # 5. Save the Output to a JSON file
        patient_name_slug = patient_data["personalInfo"]["fullName"].replace(" ", "_").lower()
        output_filename = f"diet_plan_{patient_name_slug}.json"
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(diet_plan, f, ensure_ascii=False, indent=4)
            
        # -- CHANGED LINES --
        # Replaced emojis with plain text for universal compatibility.
        print(f"Success: Successfully generated diet plan!")
        print(f"   Output saved to: {output_filename}")

    except (FileNotFoundError, ValueError, KeyError) as e:
        # -- CHANGED LINES --
        print(f"Error: An error occurred: {e}")