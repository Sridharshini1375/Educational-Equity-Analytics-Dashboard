# Educational Equity Analytics Dashboard

## Project Description
The Educational Equity Analytics Dashboard is a data-driven decision support system designed to assist policymakers in optimizing budget allocation for schools.

This project addresses the challenge of educational disparity in Tamil Nadu by utilizing machine learning to predict school performance based on infrastructure, staffing, and socioeconomic data. It provides an interactive interface for performing "what-if" analyses, allowing stakeholders to visualize the potential impact of various intervention strategies (e.g., infrastructure grants, teacher training) on student outcomes.

## Key Capabilities
* **Geospatial Analysis:** Visualizes at-risk districts and schools across Tamil Nadu using interactive maps to identify regional disparities.
* **Predictive Modeling:** Utilizes an XGBoost regression model to forecast standardized test scores based on complex, non-linear feature interactions.
* **Budget Optimization:** Implements algorithmic strategies to allocate limited financial resources, supporting both "ROI-First" (efficiency) and "Equity-First" (fairness) approaches.
* **Automated Reporting:** Generates downloadable PDF strategic plans and CSV datasets for external reporting.

## Dashboard Preview
<img width="959" height="502" alt="ss1" src="https://github.com/user-attachments/assets/8a884c50-6e39-4dac-8516-9d5a22a76990" />
<img width="676" height="389" alt="ss2" src="https://github.com/user-attachments/assets/64ff2164-c2ee-41c2-9ce1-7093399df8ab" />
<img width="671" height="395" alt="ss3" src="https://github.com/user-attachments/assets/e21a4696-d1c2-4633-97a1-a254b834f8c9" />

## Technical Architecture
The application is built using Python and consists of three core components:

1.  **Data Simulation Engine:** Generates realistic synthetic datasets representing school infrastructure and demographics in Tamil Nadu districts.
2.  **Machine Learning Pipeline:** A regression model trained on the synthetic data to identify key drivers of school performance.
3.  **Visualization Interface:** A Streamlit-based dashboard providing real-time simulation and geospatial mapping.

### Tech Stack
* **Language:** Python 3.x
* **Dashboard:** Streamlit
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Folium (Geospatial), Matplotlib
* **Reporting:** ReportLab

## Project Structure
* `dashboard_app.py`: The main entry point for the Streamlit dashboard application.
* `train_ai_planner.py`: Script to generate synthetic data, train the XGBoost model, and save the model artifacts.
* `data_utils.py`: Helper functions for data generation and feature engineering.
* `requirements.txt`: List of Python dependencies.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Educational-Equity-Analytics-Dashboard.git](https://github.com/YOUR_USERNAME/Educational-Equity-Analytics-Dashboard.git)
cd Educational-Equity-Analytics-Dashboard

```

### 2. Install Dependencies

Ensure you have Python installed. Install the required libraries using pip:

```bash
pip install -r requirements.txt

```

### 3. Initialize the Model

**Important:** You must run this script first. It generates the synthetic dataset and trains the predictive model required for the dashboard to function.

```bash
python train_ai_planner.py

```

*Output: This will create `school_master_dataset.csv`, `risk_predictor_model.joblib`, and `feature_importances.json`.*

### 4. Run the Dashboard

Launch the application locally:

```bash
streamlit run dashboard_app.py

```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Usage Guide

1. **Sidebar Configuration:** Select a specific district (e.g., Chennai, Madurai), set the annual budget, and define the risk threshold.
2. **Strategy Selection:** Choose between "ROI-First" (prioritizing score gain per rupee) or "Equity-First" (prioritizing the lowest-performing schools).
3. **Simulation:** The dashboard will forecast performance over the selected timeframe (1-15 years).
4. **Analysis:** Use the "Geospatial Analysis" tab to view regional impact and the "Year-by-Year Plan" tab to review specific budget allocations.
