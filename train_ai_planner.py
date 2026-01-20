import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
import joblib
import json

from data_utils import (
    generate_raw_data, 
    preprocess_and_engineer_features, 
    calculate_test_scores, 
    DISTRICT_COORDINATES
)

NUM_SCHOOLS_PER_DISTRICT = 100

if __name__ == '__main__':
    print(f"Generating training dataset for {len(DISTRICT_COORDINATES)} districts...")
    all_districts_df = []
    for i, district_name in enumerate(DISTRICT_COORDINATES.keys()):
        profile, facility, teacher = generate_raw_data(NUM_SCHOOLS_PER_DISTRICT, district_name)
        unique_id_offset = i * (NUM_SCHOOLS_PER_DISTRICT + 100)
        profile['pseudocode'] += unique_id_offset
        facility['pseudocode'] += unique_id_offset
        teacher['pseudocode'] += unique_id_offset
        
        district_master_df = preprocess_and_engineer_features(profile, facility, teacher)
        all_districts_df.append(district_master_df)
        
    master_df = pd.concat(all_districts_df, ignore_index=True)
    master_df = calculate_test_scores(master_df)
    
    master_df.to_csv("school_master_dataset.csv", index=False)
    print(f" Master dataset for {len(master_df)} schools saved.")
    
    print("\nTraining the master AI prediction model...")
    target = 'standardized_test_scores'
    
    features = [
        'management', 'total_class_rooms', 'total_tch', 'none_prof_qual', 
        'avg_teacher_experience_years', 'unqualified_teacher_ratio', 
        'funding_per_student', 'latitude', 'longitude',
        'building_status', 
        'electricity_availability', 
        'library_availability',
        'technology_access',
        'sanitation_quality'
    ]
    
    W_BLDG = 0.30; W_ELEC = 0.20; W_LIB = 0.20; W_TECH = 0.15; W_SANIT = 0.15
    master_df['infrastructure_score'] = 3.0 * (
        (master_df['building_status'] * W_BLDG) + 
        (master_df['electricity_availability'] * W_ELEC) +
        (master_df['library_availability'] * W_LIB) +
        (master_df['technology_access'] * W_TECH) +
        (master_df['sanitation_quality'] * W_SANIT)
    )
    features.append('infrastructure_score')
    
    features = [f for f in features if f in master_df.columns]
    
    X = master_df[features]
    y = master_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} schools, validating on {len(X_test)} schools...")
    
    model = xgb.XGBRegressor(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.05, 
        objective='reg:squarederror', 
        random_state=42
    )
    
    print("Training model (standard fit)...")
    model.fit(X_train, y_train)
    
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f" AI planner model trained. Validation Error: {test_mae:.2f} points.")

    # --- Run 5-Fold Cross-Validation ---
    print("\nRunning 5-fold cross-validation on full dataset...")
    cv_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05, 
        objective='reg:squarederror', random_state=42
    )
    
    cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f" 5-fold CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    joblib.dump(model, "risk_predictor_model.joblib")
    print(" Risk prediction model saved.")
    
    importances = pd.Series(model.feature_importances_, index=model.get_booster().feature_names).sort_values(ascending=False).to_dict()
    with open('feature_importances.json', 'w') as f:
        json.dump(importances, f, indent=4)
    print(" Feature importances saved.")
    print("\n AI training complete.")
