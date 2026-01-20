import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import folium
from streamlit_folium import st_folium
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import copy

from data_utils import DISTRICT_COORDINATES

INTERVENTIONS = {
    "Targeted Teacher Workshop": { "cost": 400000, "feature_impacted": "avg_teacher_experience_years", "improvement": 1.0, "duration": 5},
    "Full-Scale Mentorship Program": { "cost": 800000, "feature_impacted": "avg_teacher_experience_years", "improvement": 2.0, "duration": 6},
    "Technology Grant (Tablets)": { "cost": 700000, "feature_impacted": "funding_per_student", "improvement": 300, "duration": 3},
    "Strategic Funding Boost": { "cost": 1000000, "feature_impacted": "funding_per_student", "improvement": 600, "duration": 4},
    "Minor Infrastructure Repair": { "cost": 900000, "feature_impacted": "infrastructure_score", "improvement": 0.5, "duration": 8},
    "Major Infrastructure Grant": { "cost": 1500000, "feature_impacted": "infrastructure_score", "improvement": 1.2, "duration": 10}
}

def run_what_if_optimizer(at_risk_schools, budget, predictor, strategy):
    investments = []
    features = predictor.get_booster().feature_names
    
    if at_risk_schools.empty:
        return []


    LAMBDA = 0.01  # Weight for predicted score in ROI-First 
    GAMMA = 0.001  # Weight for ROI in Equity-First 
    DELTA = 0.5    # Weight for Need Index in Equity-First 
    
    need_map = {'Rural': 1.0, 'Semi-Urban': 0.5, 'Urban': 0.1}

    # 1. Calculate ROI and other metrics for all possible interventions
    for idx, school in at_risk_schools.iterrows():
        original_score = school['predicted_score']
        
        # Get Need Index 
        school_need_index = need_map.get(school.get('urban_rural', 'Urban'), 0.1)
        
        for name, details in INTERVENTIONS.items():
            cost = details['cost']
            hypothetical_school = school.copy()
            hypothetical_school[details['feature_impacted']] += details['improvement']
            
            if details['feature_impacted'] == 'infrastructure_score':
                hypothetical_school['infrastructure_score'] = min(hypothetical_school['infrastructure_score'], 3.0)
                
            hypothetical_score = predictor.predict(pd.DataFrame([hypothetical_school])[features])[0]
            estimated_score_improvement = max(0.01, hypothetical_score - original_score)
            roi = estimated_score_improvement / cost if cost > 0 else 0
            
            investments.append({
                'school_id': school['pseudocode'],
                'intervention': name,
                'roi': roi,
                'cost': cost,
                'estimated_gain': estimated_score_improvement,
                'original_score': original_score,
                'hypothetical_score': hypothetical_score,
                'need_index': school_need_index
            })

    # 2. Apply the selected strategy to prioritize 
    for inv in investments:
        if strategy == "ROI-First (Efficient)":
            # P = ROI + lambda * PredictedScore(S, I)
            inv['priority_score'] = inv['roi'] + (LAMBDA * inv['hypothetical_score'])
        else: # "Equity-First (Fairness)"
            # P = 1/Score + gamma*ROI + delta*NeedIndex
            inv['priority_score'] = (1.0 / inv['original_score']) + (inv['roi'] * GAMMA) + (inv['need_index'] * DELTA)

    # 3. Sort by the new, calculated priority score
    investments.sort(key=lambda x: x['priority_score'], reverse=True)

    # 4. Allocate budget based on the prioritized list
    final_plan, remaining_budget, schools_already_helped = [], budget, set()
    for inv in investments:
        if remaining_budget >= inv['cost'] and inv['school_id'] not in schools_already_helped:
            final_plan.append(inv)
            remaining_budget -= inv['cost']
            schools_already_helped.add(inv['school_id'])
            
    return final_plan

def create_pdf_report(history, district_name, budget, threshold, strategy):
    buffer = BytesIO(); doc = SimpleDocTemplate(buffer); styles = getSampleStyleSheet()
    story = [Paragraph("Educational Equity Strategic Plan", styles['h1']), Spacer(1, 12)]
    story.append(Paragraph("1. Executive Summary", styles['h2'])); story.append(Paragraph(f"This report outlines a {len(history)-1}-year strategic plan for '{district_name}', based on an annual budget of Rs. {budget:,}, a risk threshold of {threshold} points, and an <b>'{strategy}'</b> optimization strategy.", styles['Normal'])); story.append(PageBreak()); story.append(Paragraph(f"2. AI-Recommended {len(history)-1}-Year Strategic Plan", styles['h2']))
    for year_data in history:
        if year_data['year'] == 0: continue
        story.append(Spacer(1, 12)); story.append(Paragraph(f"<b>Year {year_data['year']}:</b>", styles['Normal']))
        if not year_data['plan']: story.append(Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;- No interventions recommended.", styles['Normal']))
        else:
            for p in year_data['plan']: line = f"&nbsp;&nbsp;&nbsp;&nbsp;- School {int(p['school_id'])}: Recommended '{p['intervention']}' (Cost: Rs. {p['cost']:,})"; story.append(Paragraph(line, styles['Normal']))
    doc.build(story); return buffer.getvalue()
def create_performance_chart(history, risk_threshold):
    years = [h['year'] for h in history]; avg_scores = [h['dataframe']['predicted_score'].mean() for h in history]; min_scores = [h['dataframe']['predicted_score'].min() for h in history]; num_at_risk = [(h['dataframe']['predicted_score'] < risk_threshold).sum() for h in history]
    fig, ax1 = plt.subplots(figsize=(10, 5)); ax1.plot(years, avg_scores, 'b-', marker='o', label='Average Score'); ax1.plot(years, min_scores, 'g--', marker='o', label='Minimum Score'); ax1.set_xlabel('Year'); ax1.set_ylabel('Predicted Score', color='b'); ax1.tick_params('y', colors='b'); ax1.grid(True); ax2 = ax1.twinx(); ax2.plot(years, num_at_risk, 'r-', marker='x', label='Number of At-Risk Schools'); ax2.set_ylabel('# At-Risk Schools', color='r'); ax2.tick_params('y', colors='r'); fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3); fig.tight_layout()
    return fig

def create_map(df, risk_threshold, map_title):
    st.subheader(map_title)
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=8 if len(df['district_name'].unique()) > 1 else 10)
    for _, school in df.iterrows():
        color = 'red' if school['predicted_score'] < risk_threshold else 'green'
        folium.CircleMarker(location=[school['latitude'], school['longitude']], radius=4, color=color, fill=True, fill_color=color, tooltip=f"School ID: {int(school['pseudocode'])}<br>Risk Score: {school['predicted_score']:.1f}").add_to(m)
    
    st_folium(m, width=700, height=400) 

st.set_page_config(layout="wide"); st.title(" Educational Equity Strategic Planner")
@st.cache_resource
def load_models_and_data():
    try:
        predictor = joblib.load("risk_predictor_model.joblib"); master_df = pd.read_csv("school_master_dataset.csv");
        with open('feature_importances.json', 'r') as f: importances = json.load(f)
        return predictor, master_df, importances
    except FileNotFoundError: return None, None, None
predictor, master_df, importances = load_models_and_data()
if predictor is None: st.error("Model or data files not found. Please run `train_ai_planner.py` first."); st.stop()

st.sidebar.title("Settings & Strategy")
selected_district = st.sidebar.selectbox("Choose a district:", ["All Districts"] + list(DISTRICT_COORDINATES.keys()))
budget_lakhs = st.sidebar.slider("Annual Budget (Lakhs INR)", 10, 500, 150); annual_budget = budget_lakhs * 100000
risk_threshold = st.sidebar.slider("Risk Threshold (Score)", 30, 90, 65)

sim_years = st.sidebar.slider("Simulation Horizon (Years)", 1, 15, 10)

strategy_selection = st.sidebar.selectbox(
    "Optimization Strategy:",
    ["ROI-First (Efficient)", "Equity-First (Fairness)"],
    help="**ROI-First:** Prioritizes interventions with the highest score-gain per rupee. **Equity-First:** Prioritizes helping the schools with the absolute lowest scores first."
)

@st.cache_data
def run_full_simulation(district_name, budget, threshold, strategy, num_years, _predictor, _master_df):
    if district_name == "All Districts":
        base_df = _master_df.copy()
    else:
        base_df = _master_df[_master_df['district_name'] == district_name].copy()
    
    features = _predictor.get_booster().feature_names
    
    base_df['predicted_score'] = _predictor.predict(base_df[features])
    
    master_upgrade_list = []
    
    simulation_history = []
    
    # Year 0 - The initial state
    simulation_history.append({'year': 0, 'plan': [], 'dataframe': base_df.copy()})

    for year in range(1, num_years + 1):
        # 1. Create the 'current_state_df' for this year
        # Start with the original data
        current_state_df = base_df.copy()
        
        # Find all upgrades that are still active
        active_upgrades = [u for u in master_upgrade_list if u['decay_year'] > year]
        
        # Apply all active upgrades to the current_state_df
        for upgrade in active_upgrades:
            try:
                idx = current_state_df[current_state_df['pseudocode'] == upgrade['school_id']].index[0]
                current_state_df.loc[idx, upgrade['feature']] += upgrade['amount']
                
                # Clamp infrastructure score to max of 3
                if upgrade['feature'] == 'infrastructure_score':
                    current_state_df.loc[idx, upgrade['feature']] = min(current_state_df.loc[idx, upgrade['feature']], 3.0)
            except IndexError:
                pass # School might not be in the selected district, safe to ignore

        # 2. Predict scores for the *current state*
        current_state_df['predicted_score'] = _predictor.predict(current_state_df[features])
        
        # 3. Find at-risk schools based on this year's state
        at_risk_schools = current_state_df[current_state_df['predicted_score'] < threshold].copy()
        
        # 4. Run the optimizer to get the plan for *this* year
        yearly_plan = run_what_if_optimizer(at_risk_schools, budget, _predictor, strategy)
        
        # 5. Add this year's plan to the master_upgrade_list
        for inv in yearly_plan:
            intervention_details = INTERVENTIONS[inv['intervention']]
            master_upgrade_list.append({
                'school_id': inv['school_id'],
                'feature': intervention_details['feature_impacted'],
                'amount': intervention_details['improvement'],
                'decay_year': year + intervention_details['duration'] 
            })
            
        # 6. Save the state *for this year* to history
        simulation_history.append({
            'year': year, 
            'plan': yearly_plan, 
            'dataframe': current_state_df.copy()
        })
        
    return simulation_history

# --- Run simulation with the new parameters ---
history = run_full_simulation(selected_district, annual_budget, risk_threshold, strategy_selection, sim_years, predictor, master_df)
initial_df = history[0]['dataframe']; final_df = history[-1]['dataframe']

st.header(f"Analysis for: {selected_district}"); st.caption(f"Using an Annual Budget of **₹{annual_budget:,}**, a Risk Threshold of **{risk_threshold} pts**, and an **'{strategy_selection}'** strategy over **{sim_years} years**.")
col1, col2, col3, col4 = st.columns(4); initial_at_risk = (initial_df['predicted_score'] < risk_threshold).sum(); final_at_risk = (final_df['predicted_score'] < risk_threshold).sum(); avg_score_change = final_df['predicted_score'].mean() - initial_df['predicted_score'].mean()
score_help_text = "This score is predicted by a pre-trained XGBoost model. The model has been trained on non-linear data, validated with 5-fold cross-validation."
col1.metric("Total Schools", f"{len(initial_df)}"); col2.metric("Initial At-Risk Schools", f"{initial_at_risk}", help="Number of schools below the threshold at Year 0."); col3.metric(f"Final At-Risk Schools (Yr {sim_years})", f"{final_at_risk}", delta=f"{final_at_risk - initial_at_risk} Schools"); col4.metric("Avg. Score Gain", f"{avg_score_change:+.2f} pts", help=score_help_text)

st.sidebar.header("Downloads"); pdf_data = create_pdf_report(history, selected_district, annual_budget, risk_threshold, strategy_selection); st.sidebar.download_button(label="Download Plan (PDF) 📄", data=pdf_data, file_name=f"Report_{selected_district}.pdf"); csv_data = initial_df.to_csv(index=False).encode('utf-8'); st.sidebar.download_button(label="Download Data (CSV) 📊", data=csv_data, file_name=f"Data_{selected_district}.csv")
tab1, tab2, tab3 = st.tabs([" Performance Forecast", " Geospatial Analysis", " Year-by-Year Plan"])
with tab1:
    st.pyplot(create_performance_chart(history, risk_threshold))
    st.subheader("AI Explainability"); st.info(f"The AI model was trained on complex, non-linear data. The top 3 factors it found to be most important for predicting scores are:")
    try:
        top_3_features = list(importances.keys())[:3]; st.code(f"1. {top_3_features[0]}\n2. {top_3_features[1]}\n3. {top_3_features[2]}")
    except (IndexError, AttributeError):
        st.code("Could not load feature importances.")
with tab2:
    col_before, col_after = st.columns(2)
    with col_before: create_map(initial_df, risk_threshold, "Before: Year 1")
    with col_after: create_map(final_df, risk_threshold, f"After: Year {sim_years}")
with tab3:
    for year_data in history[1:]: # Skip Year 0
        with st.expander(f"Year {year_data['year']} Plan & Budget"):
            plan_df = pd.DataFrame(year_data['plan'])
            if not plan_df.empty:
                st.write(f"**Budget Spent:** ₹{plan_df['cost'].sum():,} / ₹{annual_budget:,}")
                st.dataframe(plan_df[['school_id', 'intervention', 'cost', 'priority_score', 'estimated_gain']]
                             .style.format({'priority_score': "{:.6f}", 'estimated_gain': "{:.2f}"}))
            else: st.write("No interventions needed for this year.")
