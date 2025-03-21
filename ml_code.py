from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)

food_df = pd.read_csv("food_data_final.csv")

scaler = StandardScaler()
nutritional_features = ["Calories", "Protein", "Carbs", "Fats"]
food_df[nutritional_features] = scaler.fit_transform(food_df[nutritional_features])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
food_df["Cluster"] = kmeans.fit_predict(food_df[nutritional_features])

food_df[nutritional_features] = scaler.inverse_transform(food_df[nutritional_features])


def calculate_tdee(weight, height, age, activity_level, goal):
    bmr = 10 * weight + 6.25 * height - 5 * age + 5
    activity_multipliers = {
        "sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very active": 1.9,
        "extremely active": 2.0, "athlete": 2.2, "bodybuilder": 2.4
    }
    goals = {
        "weight loss": -500, "muscle gain": 500, "maintenance": 0,
        "aggressive weight loss": -750, "lean bulk": 250, "extreme bulk": 750
    }
    tdee = bmr * activity_multipliers.get(activity_level, 1.2)
    return tdee + goals.get(goal, 0)
def recommend_meals(weight, height, age, activity_level, goal, diet_preference):
    tdee = calculate_tdee(weight, height, age, activity_level, goal)
    daily_calories = tdee // 3

    filtered_df = food_df if diet_preference == "Non-Veg" else food_df[food_df["Veg/Non-Veg"] == "Veg"]
    
    avg_cluster_calories = filtered_df.groupby("Cluster")["Calories"].mean()
    if avg_cluster_calories.empty:
        return "No meal recommendations available for this diet preference."

    best_cluster = (avg_cluster_calories - daily_calories).abs().idxmin()

    def get_meal(meal_type, cluster_label):
        cluster_foods = filtered_df[(filtered_df["Cluster"] == cluster_label) & (filtered_df["Meal"] == meal_type)]
        base_foods = cluster_foods[cluster_foods["Type"] == "Base"]
        if base_foods.empty:
            base_foods = filtered_df[(filtered_df["Meal"] == meal_type) & (filtered_df["Type"] == "Base")]
        base_food = base_foods.sample(n=1)
        side_foods = cluster_foods[cluster_foods["Type"] == "Side"]
        if side_foods.empty:
            side_foods = filtered_df[(filtered_df["Meal"] == meal_type) & (filtered_df["Type"] == "Side")]
        side_foods = side_foods.sample(n=min(2, max(1, len(side_foods))))
        meal_options = pd.concat([base_food, side_foods])
        return meal_options["Food"].tolist(), meal_options["Calories"].sum(), meal_options["Protein"].sum(), meal_options["Carbs"].sum(), meal_options["Fats"].sum()
    breakfast, b_cal, b_pro, b_carb, b_fat = get_meal("Breakfast", best_cluster)
    lunch, l_cal, l_pro, l_carb, l_fat = get_meal("Lunch", best_cluster)
    dinner, d_cal, d_pro, d_carb, d_fat = get_meal("Dinner", best_cluster)
    return f"Meal Plan ({diet_preference}):\n\n" \
           f"Breakfast: {', '.join(breakfast)} ({b_cal:.0f} kcal, {b_pro:.0f}g Protein, {b_carb:.0f}g Carbs, {b_fat:.0f}g Fats)\n\n" \
           f"Lunch: {', '.join(lunch)} ({l_cal:.0f} kcal, {l_pro:.0f}g Protein, {l_carb:.0f}g Carbs, {l_fat:.0f}g Fats)\n\n" \
           f"Dinner: {', '.join(dinner)} ({d_cal:.0f} kcal, {d_pro:.0f}g Protein, {d_carb:.0f}g Carbs, {d_fat:.0f}g Fats)\n"

@app.route("/", methods=["GET", "POST"])
def index():
    meal_plan = None
    if request.method == "POST":
        weight = int(request.form["weight"])
        height = int(request.form["height"])
        age = int(request.form["age"])
        diet_preference = request.form["diet_preference"]
        activity_level = request.form["activity_level"]
        goal = request.form["goal"]

        meal_plan = recommend_meals(weight, height, age, activity_level, goal, diet_preference)

    return render_template_string("""
     <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meal Planner</title>
    <style>
        /* Light Mode */
body {
    font-family: 'Montserrat', sans-serif;
    background: linear-gradient(to right, #008080, #48bfe3);
    text-align: center;
    margin: 0;
    padding: 0;
    color: #333;
    transition: background 0.3s, color 0.3s;
}

/* Dark Mode Enhancements */
body.dark-mode {
    background: linear-gradient(to right, #0F2027, #203A43, #2C5364);
    color: #E0E0E0;
}

/* Headings */
h2 {
    margin-top: 30px;
    font-size: 2.5em;
    color: white;
    text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.3);
    animation: fadeIn 1s ease-in-out;
}

body.dark-mode h2, body.dark-mode h3 {
    color: #FFD700;
    text-shadow: 0 0 8px rgba(255, 215, 0, 0.8);
    animation: neonGlow 2s infinite alternate;
}

/* Form Container */
form {
    background: rgba(255, 255, 255, 0.9);
    width: 80%;
    max-width: 600px;
    margin: 20px auto;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.2);
    animation: fadeInUp 1s ease-in-out;
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 15px;
    align-items: center;
    border: 3px solid transparent;
    background-image: linear-gradient(white, white), linear-gradient(to right, #ff6b6b, #4ecdc4, #5d69b1, #ff9f43);
    background-clip: padding-box, border-box;
    background-origin: padding-box, border-box;
    animation: gradientBorder 5s linear infinite;
}

body.dark-mode form {
    background: rgba(30, 30, 30, 0.95);
    border: 3px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(7px);
    box-shadow: 0 0 15px rgba(255, 165, 0, 0.3);
}

/* Labels */
label {
    text-align: left;
    font-weight: 600;
    color: #555;
    display: flex;
    align-items: center;
}

body.dark-mode label {
    color: #FFB347;
}

/* Inputs & Select Fields */
input, select {
    width: 100%;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 1em;
    transition: all 0.3s ease-in-out;
}

body.dark-mode input, body.dark-mode select {
    background: #333;
    color: #FFD700;
    border: 1px solid #FFD700;
    box-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
}

body.dark-mode input:focus, body.dark-mode select:focus {
    outline: 2px solid #FFD700;
    box-shadow: 0 0 10px rgba(255, 215, 0, 0.8);
}

/* Submit Button */
input[type="submit"] {
    background: linear-gradient(135deg, #ffdd57, #ffcc00);
    color: #333;
    font-size: 1.2em;
    cursor: pointer;
    grid-column: 1 / 3;
    margin-top: 20px;
    border: none;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: background 0.3s ease, transform 0.2s ease-in-out, box-shadow 0.3s ease;
}

input[type="submit"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    background: linear-gradient(135deg, #ffcc00, #ffbb00);
}

body.dark-mode input[type="submit"] {
    background: linear-gradient(135deg, #ff8c00, #ff4500);
    color: white;
    box-shadow: 0 0 15px rgba(255, 140, 0, 0.6);
}

body.dark-mode input[type="submit"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 20px rgba(255, 140, 0, 0.8);
}

/* List Items */
ul {
    list-style-type: none;
    padding: 0;
    width: 80%;
    max-width: 600px;
    margin: 20px auto;
}

li {
    font-size: 1.1em;
    background: rgba(255, 255, 255, 0.8);
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease-in-out, background 0.3s ease-in-out;
    color: #333;
    text-align: left;
}

body.dark-mode li {
    background: rgba(40, 40, 40, 0.8);
    color: #FFD700;
    border-left: 5px solid #FFD700;
    transition: all 0.3s ease-in-out;
}

body.dark-mode li:hover {
    background: rgba(50, 50, 50, 0.9);
    transform: translateX(5px);
    box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
}

/* Toggle Switch */
.toggle-switch {
  position: relative;
  display: inline-block;
  width: 60px;
  height: 34px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #ccc;
  transition: .4s;
  border-radius: 34px;
}

.slider:before {
  position: absolute;
  content: "";
  height: 26px;
  width: 26px;
  left: 4px;
  bottom: 4px;
  background-color: white;
  transition: .4s;
  border-radius: 50%;
}

.toggle-switch input:checked + .slider {
  background-color: #FFB347;
}

.toggle-switch input:checked + .slider:before {
  background-color: #FFD700;
  transform: translateX(26px);
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes neonGlow {
    0% { text-shadow: 0 0 8px rgba(255, 215, 0, 0.7); }
    50% { text-shadow: 0 0 12px rgba(255, 215, 0, 1); }
    100% { text-shadow: 0 0 8px rgba(255, 215, 0, 0.7); }
}

@keyframes gradientBorder {
    0% { background-position: 0 0; }
    100% { background-position: 100% 0; }
}

@media (max-width: 768px) {
    form {
        grid-template-columns: 1fr;
    }

    input[type="submit"] {
        grid-column: 1;
    }
}

    </style>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap" rel="stylesheet">
</head>
<body>

    <h2>🍽️ Personalized Meal Planner</h2>

    <div class="toggle-container">
        <button class="toggle" onclick="toggleDarkMode()">🌙 Toggle Dark Mode</button>
    </div>

    <form method="POST">
        <label>Weight (kg):</label>
        <input type="number" name="weight" required value="{{ request.form.get('weight', '') }}">

        <label>Height (cm):</label>
        <input type="number" name="height" required value="{{ request.form.get('height', '') }}">

        <label>Age:</label>
        <input type="number" name="age" required value="{{ request.form.get('age', '') }}">

        <label>Diet Preference:</label>
        <select name="diet_preference">
            <option value="Veg">Vegetarian</option>
            <option value="Non-Veg">Non-Vegetarian</option>
        </select>

        <label>Activity Level:</label>
        <select name="activity_level">
            <option value="sedentary">Sedentary</option>
            <option value="light">Light</option>
            <option value="moderate">Moderate</option>
            <option value="active">Active</option>
            <option value="very active">Very Active</option>
            <option value="extremely active">Extremely Active</option>
            <option value="athlete">Athlete</option>
            <option value="bodybuilder">Bodybuilder</option>
        </select>

        <label>Goal:</label>
        <select name="goal">
            <option value="weight loss">Weight Loss</option>
            <option value="muscle gain">Muscle Gain</option>
            <option value="maintenance">Maintenance</option>
            <option value="aggressive weight loss">Aggressive Weight Loss</option>
            <option value="lean bulk">Lean Bulk</option>
            <option value="extreme bulk">Extreme Bulk</option>
        </select>

        <input type="submit" value="Get Meal Plan">
    </form>

    {% if meal_plan %}
    <h3>Your Meal Plan 🍽️</h3>
    <ul>
        {% for meal in meal_plan.split("\n") %}
            {% if meal.strip() != "" %}
                <li>{{ meal }}</li>
            {% endif %}
        {% endfor %}
    </ul>
    {% endif %}

    <script>
        function toggleDarkMode() {
            document.body.classList.toggle('dark-mode');

            // Save the mode in localStorage
            if (document.body.classList.contains('dark-mode')) {
                localStorage.setItem('darkMode', 'enabled');
            } else {
                localStorage.setItem('darkMode', 'disabled');
            }
        }

        // Load dark mode preference
        if (localStorage.getItem('darkMode') === 'enabled') {
            document.body.classList.add('dark-mode');
        }
    </script>

</body>
</html>


    """, meal_plan=meal_plan)


if __name__ == "__main__":
    app.run(debug=True)
