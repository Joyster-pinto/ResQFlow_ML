# ResQFlow ML Model Factors & Condition Handling

This document provides a comprehensive breakdown of the factors used by the ResQFlow Machine Learning models (LightGBM, CatBoost, and GNN) to predict the optimal routing path. It also details how qualitative real-world conditions (like rain, fog, and accidents) are mathematically translated into physical features the ML understands.

## 1. What the Model Predicts
**Target Variable:** `travel_time` (in seconds).
The ML pipeline's ultimate goal is to predict exactly how long it will take to traverse a single road segment (edge) under the current environmental and traffic conditions. The path with the lowest cumulative travel time is chosen as the optimal route.

---

## 2. The Features Used for Prediction

The Machine Learning ensemble looks at an array of features spanning real-time dynamics, static physical traits, and environmental inputs.

### 2.1 Live Traffic Factors
These metrics evaluate the real-time flow of traffic on a specific road edge.
*   **`vehicle_count`**: The number of vehicles currently on the road segment.
*   **`mean_speed`**: The average speed (m/s) of moving vehicles on the segment.
*   **`occupancy`**: The percentage (0.0 to 1.0) of the physical road length occupied by vehicles.
*   **`waiting_time`**: The cumulative time (seconds) vehicles have spent stopped.

### 2.2 Static Physical Factors
These metrics define the unchangeable properties of the road itself.
*   **`edge_length`**: Total physical distance of the road (in meters).
*   **`max_speed`**: The legal speed limit of the road (in m/s).
*   **`lane_count`**: The number of available traffic lanes.
*   **`road_type_code`**: *(Used natively by CatBoost)* Categorizes the road type (Residential=0, Local=1, Arterial=2, Highway=3) based on its maximum speed limit.

### 2.3 Engineered Factors 
These metrics are calculated mathematically before being handed to the model to help the models understand relationships better.
*   **`speed_ratio`**: (`mean_speed` / `max_speed`) Indicates how freely traffic is moving relative to the speed limit. A low ratio indicates congestion.
*   **`density`**: (`vehicle_count` / `edge_length`) Indicates physical overcrowding. How packed the cars are on the segment.

### 2.4 Environmental & Temporal Features
*   **`hour_of_day`**: The current hour (0-23) to capture daily traffic patterns.
*   **`is_peak_hour`**: Binary (1 or 0). 1 means it is rush hour (7–9 AM or 5–8 PM).
*   **`weather_factor`**: A numeric scaler representing the weather condition `0.0` (Clear), `0.5` (Rain), or `1.0` (Fog).

---

## 3. How Conditions Tweak the Features

The ResQFlow routing engine doesn't just hand categorical words like "Accident" to the ML. Instead, it mathematically translates those qualitative conditions into **physical consequences** (slowing speeds, increasing waiting times, etc.) which the ML model evaluates objectively to spike travel time.

### 3.1 Weather Modifiers
When the weather changes, it impacts the road's physical efficiency. This influences the simulated `mean_speed` and sets the `weather_factor`.
*   **Clear:** Speed is unhindered (1.0x). `weather_factor = 0.0`
*   **Rain:** Reduces average speed by 25% (0.75x multiplier). `weather_factor = 0.5`
*   **Fog:** Reduces average speed by 40% (0.60x multiplier). `weather_factor = 1.0`

### 3.2 Time & Day Modifiers
The time of day influences ambient background traffic volume when real-time simulation data is sparse.
*   **Rush Hour:** Automatically sets `is_peak_hour = 1`. In absence of live data, implies heavy traffic: `vehicle_count = 18`, `occupancy = 0.55`, `waiting_time = 35.0s`.
*   **Off-Peak Hour:** Sets `is_peak_hour = 0`. Implies light traffic: `vehicle_count = 2`, `occupancy = 0.05`, `waiting_time = 1.5s`.
*   **Weekend vs Weekday:** If the day is a "weekend", traffic metrics (like vehicle count and wait times) are forcefully scaled down by **55%** (a 0.45x multiplier), effectively overriding rush hour congestion logic.

### 3.3 Hazard / Incident Modifiers
If an incident is declared on the map, it mathematically degrades the road's health via severe multipliers before the ML predicts the final time.
*   **Accident:**
    *   **Speed:** Reduced by 50% (`0.50x` multiplier)
    *   **Wait Time:** Shock penalty of `+90` seconds added
    *   **Vehicle Count:** Inflated by `2.5x` (simulating instant queues)
*   **Construction:**
    *   **Speed:** Reduced by 45% (`0.55x` multiplier)
    *   **Wait Time:** Penalty of `+40` seconds added
    *   **Vehicle Count:** Inflated by `1.8x`
*   **Flood:**
    *   **Speed:** Severely reduced by 65% (`0.35x` multiplier)
    *   **Wait Time:** Penalty of `+20` seconds added
    *   **Vehicle Count:** Slightly inflated by `1.2x` (drivers avoid or slow down drastically)

### Conclusion
By mapping conditions to physical constraints rather than relying solely on categorical text inputs (e.g., Accident=True), the ML model remains robust, logical, and highly adaptable. It naturally avoids routes with accidents because the modified numbers (low speed + massive wait times + huge density) definitively compute to disastrous travel times.
