# ResQFlow: ML-Based Emergency Vehicle Routing System
## Project Report

### 1. Introduction
This report provides a comprehensive overview of the ResQFlow traffic routing system. The primary objective of this project is to optimize routing for emergency vehicles (such as ambulances, fire trucks, and police units) by predicting live travel times over the Mysore road network. We compare traditional Shortest Path algorithms (Dijkstra) against an advanced Machine Learning (ML) ensemble.

### 2. SUMO Simulation Engine & Environment
The core testing and live routing integration takes place within Eclipse SUMO (Simulation of Urban MObility):
*   **Active Map Integration**: The entire physical geometry, edge coordinates, road capacities, intersections, and traffic light logic of the city of Mysore is embedded natively into the `mysore.net.xml` SUMO network file.
*   **Vehicle Fleet Integration**: The simulation relies on individually trackable vehicle agents to test congestion. The `mysore.rou.xml` trips file introduces a baseline fleet of **721 circulating vehicles** throughout the streets.
*   **Dynamic Traffic Scaling**: During a live software sequence, our Python application bridges to SUMO via the TraCI API and actively modifies this scale factor. A "Rush Hour" setting runs SUMO with a 1.8x scale multiplier, pumping **over 1,300 active vehicles** simultaneously onto the map to physically stress-test the machine learning routing engine natively in the GUI.

### 3. Training Data Details
Because vast, live historical traffic data for Mysore is not freely available, the model relies on a state-of-the-art simulation-driven synthetic data generation process that correctly reflects real-world traffic physics:

*   **Network Generation**: Extracting real road typologies, speed limits, lengths, and intersection constraints from the Mysore SUMO Network file (`mysore.net.xml`).
*   **Traffic Simulation Engine**: Simulating different hours naturally matching peak flow vs. off-peak flow.
*   **Dataset Samples**: 10,000+ realistic edge states are sampled to train the models. The target variable is `travel_time` (the true time it takes to traverse an edge), which dynamically shifts from the base free-flow speed when heavily congested.

### 4. Machine Learning Algorithms
To ensure robust, high-performance inferences, ResQFlow uses a three-model **Ensemble Predictor**:

1.  **LightGBM Regressor:** A highly efficient gradient-boosted tree model. It is very fast at inference and serves as a reliable performance baseline.
2.  **CatBoost Regressor:** A gradient-boosting model optimized to handle categorical features structurally. CatBoost explicitly models the `road_type_code` (residential vs. local vs. arterial vs. highway) without needing one-hot encoding, capturing speed hierarchies accurately.
3.  **Graph Neural Network (PyTorch Geometric):** A Custom GraphSAGE (`RoadGNN`). Unlike tree-based algorithms which look at one road at a time, the GNN predicts travel times by looking at the *entire neighborhood topology* (understanding bottleneck propagation from surrounding roads).

The predictions from these three models are merged at runtime, guaranteeing that the routing algorithm finds the true fastest path based on context.

### 5. Prediction Factors (Model Features)
For every single stretch of road, the models take the following 13 environmental and static features to output an estimated travel time in seconds:
*   `vehicle_count`: Number of vehicles currently on the edge.
*   `mean_speed`: The average real-time speed of those vehicles.
*   `occupancy`: Congestion percentage of the lane capacity.
*   `waiting_time`: Time vehicles have spent stopped/waiting at intersections.
*   `edge_length`: Length of the road segment (meters).
*   `max_speed`: Legal speed limit (m/s).
*   `lane_count`: Throughput capacity.
*   `hour_of_day`: Current wall clock hour (0-23).
*   `is_peak_hour`: Binary flag (Rush hour: 7-9 AM, 5-7 PM).
*   `speed_ratio`: (Engineered factor) Current speed / Speed Limit.
*   `density`: (Engineered factor) Vehicles per meter.
*   `weather_factor`: Scalar representing weather degradation (0.0 Clear, 0.5 Rain, 1.0 Fog).
*   `road_type_code`: Used explicitly in CatBoost to differentiate Highways from Local streets.

### 6. Dynamic Modifiers Handled
The simulation handles highly dynamic live events that impact speed limits and traffic geometry severely:
*   **Time of Day**: "offpeak", "rush" (inflates base vehicle count by 1.8x), Weekend vs Weekday scale factors.
*   **Weather**: "clear" (1.0x base speed), "rain" (0.75x speed limit), "fog" (0.60x speed limit with higher wait times).
*   **Road Hazards**: Simulating physical road blocks:
     *   `accident`: 50% speed reduction and severe queue pileups (+90s wait time).
     *   `construction`: 55% speed reduction (+40s wait time).
     *   `flood`: Total gridlock behaviors down to 0.5 m/s.

### 7. Visual Performance Analysis
To demonstrate the model's adaptive intelligence across environments, a routing experiment was simulated over a long-distance cross-city path.
*   **Source:** K.R. Hospital
*   **Destination:** Udayagiri 

The test runs through **7 distinct environmental modifications**. The visualization below charts the travel times of a traditional shortest distance path (Dijkstra) versus our AI Ensemble path.

> [!TIP]
> **Observation**: Notice that under clear off-peak conditions, Dijkstra and the ML path time estimates are relatively parallel. However, when a Road Hazard (like an Accident) or Rush Hour traffic hits, the traditional Shortest Path tries going right through the gridlock, resulting in drastic time delays. The ML Ensemble intelligently detects the congestion and *diverts* to slightly longer roads that are physically clearer, slicing down total travel time significantly.

#### Chart: Travel Time comparison under Modifiers
![Modifier Impact Chart](file:///D:/ResQFlow_ML/data/visuals/modifier_impact.png)
