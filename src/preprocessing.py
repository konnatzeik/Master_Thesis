import pandas as pd
import numpy as np
import re

#--------------------------------------------
#                Data Loading
#--------------------------------------------

def load_data(file_path:str) -> pd.DataFrame:
    """
    Load and merge patient-level and visit-level data.
    file_path: the path to the Excel file 

    Returns a pd.DataFrame, containing static and longitudinal data.

    """

    df_visits = pd.read_excel(file_path, sheet_name='Visits_Longitudinal')
    df_static = pd.read_excel(file_path, sheet_name='Patient_Static')

    #execute a right join to keep all visit records
    df = pd.merge(df_static, df_visits, on='patient_id', how='right', validate="one_to_many")

    return df



#--------------------------------------------
#                Data Cleaning
#--------------------------------------------

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns with time information to the preffered format .

    visit_date and date_of_birth -> datetime format
    year_of_diagnosis -> integer format
    
    This function ensures that the date columns have the correct format, 
    in order to be used in feature engineering.
    """

    #convert dates to datetime format
   
    df['visit_date'] = pd.to_datetime(df['visit_date'], dayfirst=True, errors='coerce')
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], dayfirst=True, errors='coerce')

    #convert from float to integer
    if 'year_of_diagnosis' in df.columns:
        df['year_of_diagnosis'] = df['year_of_diagnosis'].astype('Int64')

    return df

def remove_duplicates(df:pd.DataFrame) -> pd.DataFrame:
    """
    Check for duplicate rows based on patient_id and visit_date, and remove them if found.
    """

    #Checking if there any duplicates and removes them
    duplicates = df.duplicated(subset=['patient_id', 'visit_date']).sum()
    if duplicates > 0:
        print(f"{duplicates} duplicate rows were founded. Removing them...")
        df = df.drop_duplicates(subset=['patient_id', 'visit_date'])
    
    return df

def handle_categorical_missing(df:pd.DataFrame) -> pd.DataFrame:

    """
    Replace missing values in categorical features with 'none' and 
    'no_switch', to distinguish between true missingness and the absence of a category.
    """

    fill_values = {'switch_reason': 'no_switch',
                     'csdmard_name': 'none',
                     'b_ts_dmard_name': 'none',
                     'comorbidities': 'none'
                    }

    for col, value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value).astype(str)

    return df

def clean_treatment_decision(df:pd.DataFrame) -> pd.DataFrame:
    """
    Clean the categorical feature 'treatment_decision'
    """

    df['treatment_decision'] = df['treatment_decision'].astype(str).str.strip().str.lower()
    df['treatment_decision'] = df['treatment_decision'].replace('nan', pd.NA)

    #fill missing values with 'no_decision' to indicate that in the baseline visits, there are no treatment decisions
    df['treatment_decision'] = df['treatment_decision'].fillna('no_decision')

    #group treatment decisions into broader categories to reduce the number of categories
    df['treatment_decision_grouped'] = df['treatment_decision'].replace({
        "escalation": "escalation_or_switch",
        "switch": "escalation_or_switch",
        "tapering": "tapering_or_stop",
        "stop": "tapering_or_stop"
    })

    return df

def encode_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features into numeric binary values.

    Static features:    gender
                        rf_status
                        anti_ccp_status

    Yes/No features:    flare_event
                        mtx_use
    """

    #Binary mappings for static features
    binary_mapping = {
        'gender': {'Female': 1, 'Male': 0},
        'rf_status': {'Positive': 1, 'Negative': 0},
        'anti_ccp_status': {'Positive': 1, 'Negative': 0}
    }

    for col, mapping in binary_mapping.items():
        if col in df.columns:
            df[col] = df[col].str.strip().map(mapping).astype('Int64')
    
    #Binary mapping for Yes/No features
    mapping = {'Yes': 1, 'No': 0}
    yes_no_columns = ['flare_event', 'mtx_use']
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(mapping).astype('Int64')
    
    return df

def clean_comorbidities(df:pd.DataFrame) -> pd.DataFrame:
    """
     Clean and standardize comorbidities column, 
     a free-text column that contains a list of comorbidities for each patient.
    """
    #normalize text format
    df['comorbidities'] = df['comorbidities'].str.lower().str.strip()

    #replace 'none' with an empty string to show the absence of comorbidities, 
    # in order to avoid confusion with missing values
    df['comorbidities'] = df['comorbidities'].str.replace(r'\bnone\b', '', regex=True) 
    
    #unify separators to commas
    df['comorbidities'] = df['comorbidities'].replace({';': ',', '/': ',', r'\band\b': ','}, regex=True)

    #replace mulitple spaces with a single space. E.g. hypertension,  dyslipidemia -> hypertension, dyslipidemia
    df['comorbidities'] = df['comorbidities'].str.replace(r'\s+', ' ', regex=True)

    #replace multiple commas with a single comma. E.g. hypertension,, dyslipidemia -> hypertension, dyslipidemia
    df['comorbidities'] = df['comorbidities'].str.replace(r',+', ',', regex=True)

    #removes commas at the beginning and the end of the string
    df['comorbidities'] = df['comorbidities'].str.strip(',')

    #remove extra spaces around commas. E.g. hypertension, dyslipidemia -> hypertension,dyslipidemia
    df['comorbidities'] = df['comorbidities'].str.replace(r'\s*,\s*', ',', regex=True)

    return df


def drop_sparse_columns(df:pd.DataFrame, threshold:float=0.5) -> pd.DataFrame:
    """
    Drop columns with a proportion of missing values greater than the given threshold.
    Keep the protected columns that are crucial for the analysis and should be kept regardless of missingness.
    """
    missing_fraction = df.isnull().mean()

    #clinical features that are important despite their missingness
    protected_cols = ['das28_score', 'crp', 'esr']

    cols_exceeding_threshold = missing_fraction[missing_fraction > threshold].index
    cols_to_drop = [col for col in cols_exceeding_threshold if col not in protected_cols]

    if len(cols_to_drop) > 0:
        print(f"Dropping columns with > {threshold:.0%} missing values: {', '.join(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)
    
    return df






#--------------------------------------------
#                Feature Engineering
#--------------------------------------------



def add_birth_year(df:pd.DataFrame) -> pd.DataFrame:
    """
    Extract only birth year from date_of_birth,
    and drop the original date_of_birth column, to avoid re-identification of patients.
    """
    
    df['birth_year'] = df['date_of_birth'].dt.year.astype('Int64')
    df.drop(columns=['date_of_birth'], inplace=True)
    
    return df


def time_intervals(df:pd.DataFrame) ->pd.DataFrame:

    """
    Create time-based longitudinal features for each visit.

    Features created:
    - days_since_last_visit: number of days since the previous visit
    - disease_duration: years since diagnosis at each visit
    - age_at_visit: patient's age in years at each visit

    Visits are first sorted chronologically within each patient.
    """
    
    # ensure chronological order
    df = df.sort_values(['patient_id', 'visit_date'])

    #calculate how many days have passed since the last visit
    #(the very first visit that has not a previous one is filled with 0)
    df['days_since_last_visit'] = (df.groupby('patient_id')['visit_date'].diff().dt.days.fillna(0)).astype('Int64')

    #calculate disease duration (in years)
    df['disease_duration'] = (df['visit_date'].dt.year - df['year_of_diagnosis']).astype('Int64')

    #calculate age at each visit
    df['age_at_visit'] = (df['visit_date'].dt.year - df['birth_year']).astype('Int64')
    
    return df


def das28_change(df:pd.DataFrame) -> pd.DataFrame:
    """
    Calculate change in DAS28 score compared to previous visit.
    Positive value: increase in disease activity
    Negative value: decrease in disease activity
    """

    df = df.sort_values(['patient_id', 'visit_date'])
    #Difference in DAS28 score compared to previous visit (fill the first visit with 0)
    df['das28_change'] = df.groupby('patient_id')['das28_score'].diff()

    #the first visit of each patient has no previous visits to compare, so it is filled with 0
    first_visit = df.groupby('patient_id').head(1).index
    df.loc[first_visit, 'das28_change'] = 0

    return df

def previous_flare(df:pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature indicating if a flare occurred at the previous visit.
    """

    df = df.sort_values(['patient_id', 'visit_date'])

    #previous flare status (fill the first visit with 0)
    df['previous_flare'] = df.groupby('patient_id')['flare_event'].shift(1).fillna(0).astype(int)

    return df

def flare_next_visit(df:pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable indicating whether a flare occurs
    at the next visit for each patient.
    """

    df = df.sort_values(['patient_id', 'visit_date'])

    #define target variable
    df['target_flare_next'] = df.groupby('patient_id')['flare_event'].shift(-1)

    #drop rows where target variable is missing (the last visit of each patient)
    df = df.dropna(subset=['target_flare_next'])
    df['target_flare_next'] = df['target_flare_next'].astype(int)

    return df


def lab_features(df:pd.DataFrame) -> pd.DataFrame:

    """
    Features created:
    -crp_missing: binary feature inidcating if the CRP value is missing
    -esr_missing: binary feature inidcating if the ESR value is missing
    -crp_normalized: CRP value is normalized by the lab-specific upper limit of normal 
    -log_crp_normalized: log-transformed CRP value
    -inflammation_flag: binary feature indicating if there is evidence of inflammation based on CRP and ESR values
     (CRP above the upper limit of normal or ESR above 20 mm/hr)
    -inflammation_score: a score from 0 to 2 indicating the degree of inflammation (0 = neither high, 1 = one high, 2 = both high)
    """
    
    df['crp_missing'] = df['crp'].isna().astype(int)
    df['esr_missing'] = df['esr'].isna().astype(int)

    #normalize CRP value to the given upper limit of normal
    df['crp_normalized'] = df['crp']/df['crp_upper_limit']
    #log-transform the normalized CRP value to reduce skewness
    df['log_crp_normalized'] = np.log(df['crp_normalized'])

    #Binary inflammation indicator
    df['crp_high'] = (df['crp_normalized'] > 1).fillna(False)
    df['esr_high'] = (df['esr'] > 20).fillna(False)
    df['inflammation_flag'] = (df['crp_high'] | df['esr_high']).astype(int)

    # 0 = neither high, 1 = one high, 2 = both high
    df['inflammation_score'] = df['crp_high'].astype(int) + df['esr_high'].astype(int)

    return df


def comorbidities_groups(df:pd.DataFrame) -> pd.DataFrame:
    """
    Group individual comorbidities into separate clinical categories, and create binary features for each group

    Create a comorbidity count representing the total number of  comorbidities for each patient.
    """

    comorbidity_mapping = {
        'comorb_cardiovascular': [
            'transient ischemic attack', 'hypertension', 'coronary artery disease', 
            'heart failure', 'atrial fibrillation', 'chronic venous insufficiency',
            'carotid stenosis', 'ventricular septal defect',
            'paroxysmal supraventricular tachycardia', 'ascending aortic aneurysm', 'pericarditis',
            'venous thrombosis', 'peripheral vascular disease', 'aortic valve stenosis',
            'pulmonary arterial hypertension', 'mitral valve disease', 'myocardial infarction'
        ],
        'comorb_metabolic_endocrine': [
            'hypothyroidism', 'hashimoto', 'dyslipidemia',
            'diabetes mellitus', 'cushing syndrome', 'morbid obesity'
        ],
        'comorb_respiratory' : [
            'chronic obstructive pulmonary disease', 'asthma',
            'pulmonary fibrosis', 'tuberculosis', 'emphysema'
        ],
        'comorb_renal_urological' : [
            'nephrolithiasis', 'prostatitis', 'benign prostatic hyperplasia',
            'solitary kidney', 'chronic kidney disease'
        ],                               
        'comorb_gastrointestinal_hepatic' : [
            'hepatitis b', 'resolved hepatitis b', 'hepatic steatosis',
            'gastroesophageal reflux disease', 'total gastrectomy', 'cholecystectomy',
            'operated perianal abscess', 'barretts esophagus', 'irritable bowel syndrome'
        ],                                     
        'comorb_malignancy' : [
            'kidney cancer', 'breast cancer', 'parotid cancer', 'operated meningioma'
        ],                        
        'comorb_musculoskeletal' : [
            'osteoporosis', 'osteopenia', 'osteoarthritis',
            'gout', 'lumbar spondylosis', 'bilateral carpal tunnel syndrome',
            'degenerative cervical spine disease'
        ],                             
        'comorb_neurologic_psychiatric' : [
            'depression', 'anxiety disorder', 'parkinsonism',
            'vertigo', 'dementia', 'postherpetic neuralgia'
        ],                                    
        'comorb_autoimmune_other' : [
            'sjogren syndrome', 'thalassemia', 'beta thalassemia minor',
            'monoclonal gammopathy of undetermined significance', 'mesenteric lipodystrophy',
            'blepharitis', 'keratitis', 'herpes zoster', 'glaucoma', 'hives'
        ]                                     
                                
    }

    for group, diseases in comorbidity_mapping.items():
        pattern = '|'.join(map(re.escape, diseases))
        df[group] = df['comorbidities'].str.contains(pattern, na=False).astype(int)

    df['comorbidity_count'] = df['comorbidities'].apply(lambda x: 0 if x == '' else len(x.split(',')))


    return df

def treatment_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    Create binary features indicating whether a patient is on csDMARDs,
    biological, targeted synthetic DMARDs, or steroids at each visit.
    Create a treatment count feature representing the total number of treatments for each visit.
    """

    #Binary features for treatment categories (0 = not on treatment, 1 = on treatment)
    df['csdmard_use'] = df['csdmard_name'].apply(lambda x: 0 if x == 'none' else 1)
    df['b_ts_dmard_use'] = df['b_ts_dmard_name'].apply(lambda x: 0 if x == 'none' else 1)
    df['steroid_use'] = (df['steroid_dose']>0).astype(int)

    #treatment count
    df['treatment_count'] = df[['csdmard_use', 'b_ts_dmard_use', 'steroid_use','mtx_use']].sum(axis=1)

    return df


#--------------------------------------------
#                Data Validation
#--------------------------------------------

def validation_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Run sanity checks:
    -negative disease duration (e.g. visit date before year of diagnosis)
    -unrealistic age (e.g. negative age, age above 110 years)
    -negative visit intervals (e.g. visit date before previous visit date)
    """

    #check if disease duration is negative due to type error
    invalid_duration = df[df['disease_duration']<0]
    if not invalid_duration.empty:
        print('Negative disease duration detected')
        print(invalid_duration[['patient_id', 'visit_date', 'year_of_diagnosis', 'disease_duration']])

    #check if age is unrealistic
    invalid_age = df[(df['age_at_visit']<0) | (df['age_at_visit']>110)]
    if not invalid_age.empty:
        print('Unrealistic age detected.')
        print(invalid_age[['patient_id', 'birth_year', 'age_at_visit', 'visit_date']]) 

    # Check for negative visit intervals
    invalid_interval = df[df['days_since_last_visit'] < 0]
    if not invalid_interval.empty:
        print("Negative days_since_last_visit detected:")
        print(invalid_interval[['patient_id', 'visit_date', 'days_since_last_visit']])

    return df

#-----------------------------------------------------
#                Preprocessing Pipeline
#-----------------------------------------------------

def preprocess_data(file_path:str) ->pd.DataFrame:

    df = load_data(file_path)

    #data cleaning
    df = convert_dates(df)
    df = remove_duplicates(df)  
    df = handle_categorical_missing(df)
    df = clean_treatment_decision(df)
    df = encode_features(df)   
    df = clean_comorbidities(df)
    df = drop_sparse_columns(df) 

    #feature engineering
    df = add_birth_year(df)
    df = time_intervals(df)   
    df = das28_change(df)
    
    df = previous_flare(df)
    df = flare_next_visit(df)
    
    df = lab_features(df)
    df = comorbidities_groups(df)
    df = treatment_features(df)

    #data validation
    df = validation_data(df)

    return df 

