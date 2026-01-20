# Educational Equity Analytics Dashboard

## Project Description
The Educational Equity Analytics Dashboard is a data-driven decision support system designed to assist policymakers in optimizing budget allocation for schools. 

This project addresses the challenge of educational disparity by utilizing machine learning to predict school performance based on infrastructure, staffing, and socioeconomic data. It provides an interactive interface for performing "what-if" analyses, allowing stakeholders to visualize the potential impact of various intervention strategies (e.g., infrastructure grants, teacher training) on student outcomes.

## Key Capabilities
* **Geospatial Analysis:** Visualizes at-risk districts and schools across Tamil Nadu using interactive maps to identify regional disparities.
* **Predictive Modeling:** Utilizes an XGBoost regression model to forecast standardized test scores based on complex, non-linear feature interactions.
* **Budget Optimization:** Implements algorithmic strategies to allocate limited financial resources, supporting both "ROI-First" (efficiency) and "Equity-First" (fairness) approaches.
* **Automated Reporting:** Generates downloadable PDF strategic plans and CSV datasets for external reporting.

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
