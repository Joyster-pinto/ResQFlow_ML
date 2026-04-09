# ResQFlow: ML-Driven Emergency Routing Simulation
## Comprehensive Project Report

### Project Overview
ResQFlow is an advanced, predictive routing system designed to optimize emergency vehicle dispatch (hospitals, fire stations, and police departments). Rather than relying solely on static map distances, ResQFlow integrates real-world traffic data, environmental conditions, and live incident hazards using an ensemble of Machine Learning (ML) models. It couples these ML predictions with microscopic traffic simulation to compute the true fastest route for first responders.

This report summarizes the architectural foundation of the project from scratch, including the training data architecture, the simulation environment, the conditional modifiers, and the visual front-end representation.

---

### 1. Data Used to Train the Model

The core intelligence of ResQFlow relies on an ensemble machine learning approach (utilizing CatBoost, LightGBM, and Graph Neural Networks). To train these models to understand traffic congestion and predict travel times, a highly detailed dataset (`traffic_samples.csv`) comprising over 5,000 synthetic traffic scenarios was used.

**Data Generation Process:**
The dataset was generated computationally via `data_generator.py`, which systematically ran thousands of routing variations within the simulation to catalog how vehicles behaved under different constraints.

**Features (Inputs):**
The models were trained on a matrix of 14 key features categorized into three groupings:
1. **Live Traffic Factors:** Real-time diagnostics pulled from the roads, including `vehicle_count`, `mean_speed`, `occupancy` (density of vehicles), and `waiting_time` (cumulative seconds vehicles spent stopped).
2. **Static Physical Factors:** The unchangeable topology of the road segments, such as `edge_length`, `max_speed` (legal limits), `lane_count`, and road categorical types.
3. **Engineered & Temporal Factors:** Calculated metrics such as the `speed_ratio` (average speed vs. speed limit), `hour_of_day`, and an `is_peak_hour` binary identifier. 

**Target Objective (Output):**
The predictive variable for the ML models is **`travel_time`** (measured in seconds). By evaluating the physical properties and the live conditions of a road, the models predict exactly how long it takes to traverse that specific segment, allowing the routing algorithm to chain segments together to find the true quickest path.

---

### 2. The SUMO Simulation Environment

To mirror real-world complexities, the project leverages **SUMO (Simulation of Urban MObility)**, an open-source, highly portable microscopic traffic simulation package. 

**Network Architecture:**
The simulation's foundational map focuses on the real-world road network of **Mysore, Karnataka**. The geographical layout was extracted using OpenStreetMap (OSM) data and compiled into rigorous XML schemas that SUMO understands:
* `mysore.net.xml`: Represents the mathematical nodes and edges of the roads.
* `mysore.poly.xml`: Defines boundary shapes, landmarks, and structural polygons.
* `mysore.rou.xml` & `mysore.trips.xml`: Govern the logic of background "ambient" civilian traffic navigating the city to generate organic congestion.

**TraCI Integration:**
ResQFlow communicates with SUMO natively via **TraCI** (Traffic Control Interface). This bi-directional Python socket allows the routing system to query the simulation live (e.g., "How many cars are currently jammed on Edge A?") and inject the emergency vehicle dynamically so it can interact with and push past the simulated civilian traffic in real-time.

---

### 3. Modifiers Used to Modify the Simulation

When computing routing logic, real-world phenomena—like sudden rain or an accident—must be mathematically translated into constraints the ML model can objectively evaluate. ResQFlow does not just pass categorical words to the ML; it uses systematic **Multipliers** to drastically degrade a road's statistical health, forcing the ML to predict terrifying travel times, thereby naturally discouraging the algorithm from using that road.

**Weather Modifiers:**
Weather downgrades physical efficiencies and influences ambient vehicle speeds:
* **Clear (`factor=0.0`):** Operates at maximum efficiency (1.0x baseline).
* **Rain (`factor=0.5`):** Curvatures and skid-risks reduce safe travel speeds by 25% (0.75x multiplier).
* **Fog (`factor=1.0`):** Severely constrained visibility reduces speeds by 40% (0.60x multiplier).

**Time & Day Modifiers:**
When TraCI live-data is sparse, temporal profiles apply simulated background noise:
* **Rush Hour (7-9 AM / 5-8 PM):** Automatically balloons traffic values (injects default vehicle counts and 35-second average wait penalties).
* **Weekend Adjustments:** Reduces ambient traffic volume expectations by 55%.

**Hazard / Incident Modifiers:**
If an incident is declared on a node, surrounding edges undergo severe mathematical penalties:
* **Accidents:** Speeds slashed by 50%, an automatic +90 second waiting-time penalty is applied, and structural density is ballooned by 2.5x to simulate instantaneous trailing gridlock.
* **Construction:** Speeds cut by 45%, +40 seconds of wait time, 1.8x density inflation.
* **Flooding:** A catastrophic 65% reduction in allowable speeds, simulating extreme navigation caution.

---

### 4. Visual Representation of ML Paths

The routing predictions and analytical results are hosted on a Flask-based web backend, visualized precisely on an interactive frontend dashboard.

**Mapping Engine (Leaflet.js):**
The user interface utilizes Leaflet.js rendering an OpenStreetMap tileset. Specifically, a dark, low-contrast, 'dull' basemap (via Stadia Maps Alidade Smooth Dark) was selected. This design choice strips away background visual noise (bright greens/yellows of default maps), which forces the brightly colored route vectors to visually "pop" and command immediate operator attention.

**Route Comparison via PolyLines:**
The dashboard allows for visual **"Apples-to-Apples" route comparison**. 
1. **Dijkstra's Path:** Represents the mathematically shortest geographical distance, typically rendered as a subdued or dashed line. It assumes ideal conditions.
2. **The ML Predicted Path:** By factoring in the modifiers (weather, accidents, traffic data), the ensemble models calculate a dynamically updated path, drawn as a solid, vibrant vector line.

By displaying both simultaneously, the interface visualizes exactly when and why the ML model diverged from the "shortest" route—for instance, choosing a longer highway detour to bypass a severe ML-identified traffic bottleneck caused by localized flooding. The interface uses customized iconography (Hospitals 🏥, Fire 🚒, Police 🚔) as drag-and-drop interactive dispatch nodes.
