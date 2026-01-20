import pandas as pd
import numpy as np

DISTRICT_COORDINATES = {
    "Chennai": {"lat_min": 12.95, "lat_max": 13.15, "lon_min": 80.18, "lon_max": 80.28},
    "Coimbatore": {"lat_min": 10.9, "lat_max": 11.1, "lon_min": 76.85, "lon_max": 77.05},
    "Madurai": {"lat_min": 9.8, "lat_max": 10.0, "lon_min": 78.0, "lon_max": 78.2},
    "Tiruchirappalli": {"lat_min": 10.7, "lat_max": 10.9, "lon_min": 78.6, "lon_max": 78.8},
    "Salem": {"lat_min": 11.55, "lat_max": 11.75, "lon_min": 78.05, "lon_max": 78.25},
    "Tirunelveli": {"lat_min": 8.6, "lat_max": 8.8, "lon_min": 77.6, "lon_max": 77.8},
    "Vellore": {"lat_min": 12.8, "lat_max": 13.0, "lon_min": 79.0, "lon_max": 79.2},
    "Erode": {"lat_min": 11.25, "lat_max": 11.45, "lon_min": 77.6, "lon_max": 77.8},
    "Thoothukudi": {"lat_min": 8.65, "lat_max": 8.85, "lon_min": 78.05, "lon_max": 78.2},
    "Villuppuram": {"lat_min": 11.8, "lat_max": 12.0, "lon_min": 79.4, "lon_max": 79.6}
}

def generate_raw_data(num_schools, district_name):
    config = DISTRICT_COORDINATES[district_name]
    socioeconomic_factor = np.random.uniform(0.1, 1.0, num_schools)
    school_ids = np.arange(1000, 1000 + num_schools)
    
    latitudes = np.random.uniform(config['lat_min'], config['lat_max'], num_schools)
    longitudes = np.random.uniform(config['lon_min'], config['lon_max'], num_schools)
    
    # --- Profile DataFrame ---
    profile_df = pd.DataFrame({
        'pseudocode': school_ids,
        'latitude': latitudes,
        'longitude': longitudes,
        'district_name': district_name
    })
    
    def assign_urban_rural(x):
        if x > 0.7: return 'Urban'
        if x > 0.4: return 'Semi-Urban'
        return 'Rural'
    profile_df['urban_rural'] = [assign_urban_rural(x) for x in socioeconomic_factor]

    profile_df['management'] = [1 if x < 0.7 else 4 for x in socioeconomic_factor]
    profile_df['grants_receipt'] = (5000 + socioeconomic_factor * 50000 + np.random.normal(0, 5000, num_schools)).round(0)
    
    # --- Facility DataFrame ---
    facility_df = pd.DataFrame({'pseudocode': school_ids})
    facility_df['building_status'] = [1 if x > 0.4 else 0 for x in socioeconomic_factor]
    facility_df['electricity_availability'] = [1 if x > 0.3 else 0 for x in socioeconomic_factor]
    facility_df['library_availability'] = [1 if x > 0.35 else 0 for x in socioeconomic_factor]
    facility_df['technology_access'] = [1 if x > 0.5 else 0 for x in socioeconomic_factor]
    facility_df['sanitation_quality'] = [1 if x > 0.25 else 0 for x in socioeconomic_factor]
    facility_df['total_class_rooms'] = (8 + (socioeconomic_factor * 12)).round(0).astype(int)

    # --- Teacher DataFrame ---
    teacher_df = pd.DataFrame({'pseudocode': school_ids})
    total_teachers = (facility_df['total_class_rooms'] / 2.2 + np.random.normal(0, 2, num_schools)).round(0).astype(int).clip(lower=1)
    teacher_df['total_tch'] = total_teachers
    teacher_df['none_prof_qual'] = (total_teachers * (1 - socioeconomic_factor) * 0.5).round(0).astype(int)
    teacher_df['avg_teacher_experience_years'] = (2 + (socioeconomic_factor * 12) + np.random.normal(0, 2, num_schools)).round(1)
    
    return profile_df, facility_df, teacher_df

def preprocess_and_engineer_features(profile_df, facility_df, teacher_df):
    master_df = pd.merge(profile_df, facility_df, on='pseudocode')
    master_df = pd.merge(master_df, teacher_df, on='pseudocode')
    
    master_df['unqualified_teacher_ratio'] = (master_df['none_prof_qual'] / master_df['total_tch']).fillna(0)
    
    W_BLDG = 0.30
    W_ELEC = 0.20
    W_LIB = 0.20
    W_TECH = 0.15
    W_SANIT = 0.15
    
    master_df['infrastructure_score'] = 3.0 * (
        (master_df['building_status'] * W_BLDG) + 
        (master_df['electricity_availability'] * W_ELEC) +
        (master_df['library_availability'] * W_LIB) +
        (master_df['technology_access'] * W_TECH) +
        (master_df['sanitation_quality'] * W_SANIT)
    )

    
    master_df['funding_per_student'] = master_df['grants_receipt'] / (master_df['total_class_rooms'] * 20)
    
    return master_df

def calculate_test_scores(df):
    """
    This is the non-linear calculation from your original code,
    which matches the claim in Section III-B (Eq. 1). It is unchanged.
    """
    base_score = 35
    
    infra_multiplier = 0.5 + (df['infrastructure_score'] / 3.0) * 0.7 
    
    teacher_effect = df['avg_teacher_experience_years'] * 1.8
    funding_effect = (df['funding_per_student'] / 500) * 5.0
    
    final_scores = base_score + (teacher_effect * infra_multiplier) + funding_effect + np.random.normal(0, 4, len(df))
    
    df['standardized_test_scores'] = np.clip(final_scores, 20, 95).round(1)
    return df
