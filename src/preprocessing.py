import pandas as pd
import re

#--------------------------------------------
#                Data Loading
#--------------------------------------------

def load_data(file_path:str) -> pd.DataFrame:
    """
    Load and merge patient-level and visit-level data
    file_path: str, the path to the Excel file 

    Returns a pd.DataFrame, containing static and longitudinal data

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
    Change date columns into datetime format
    """

    #convert data columns
    df['visit_date'] = pd.to_datetime(df['visit_date'], dayfirst=True)
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], dayfirst=True)

    #convert from float to integer
    df['year_of_diagnosis'] = df['year_of_diagnosis'].astype('Int64')

    return df

def handle_categorical_missing(df:pd.DataFrame) -> pd.DataFrame:

    if 'switch_reason' in df.columns:
        df['switch_reason'] = df['switch_reason'].fillna('no_switch')

    if 'csdmard_name' in df.columns:
        df['csdmard_name'] = df['csdmard_name'].fillna('none')

    if 'b_ts_dmard_name' in df.columns:
        df['b_ts_dmard_name'] = df['b_ts_dmard_name'].fillna('none')

    if 'comorbidities' in df.columns:
        df['comorbidities'] = df['comorbidities'].fillna('none')

    return df


def encode_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    Transform static text features into numeric binary values.
    """
    #define columns and their mappings
    binary_mapping = {
        'gender': {'Female': 1, 'Male': 0},
        'rf_status': {'Positive': 1, 'Negative': 0},
        'anti_ccp_status': {'Positive': 1, 'Negative': 0}
    }

    for col, mapping in binary_mapping.items():
        if col in df.columns:
            df[col] = df[col].str.strip().map(mapping).astype('Int64')
    
    mapping = {'Yes': 1, 'No': 0}
    if 'flare_event' in df.columns:
        df['flare_event'] = df['flare_event'].str.strip().map(mapping).astype('Int64')

    if 'mtx_use' in df.columns:
        df['mtx_use'] = df['mtx_use'].str.strip().map(mapping).astype('Int64')
    
    return df

def comorbidity_clean(df:pd.DataFrame) -> pd.DataFrame:
     
    df['comorbidities'] = df['comorbidities'].str.replace(r'\bnone\b', '', regex=True) 
    df['comorbidities'] = df['comorbidities'].str.lower().str.strip()
    df['comorbidities'] = df['comorbidities'].replace({';': ',', '/': ',', ' and ': ','}, regex=True)
    df['comorbidities'] = df['comorbidities'].str.replace(r'\s+', ' ', regex=True)
    df['comorbidities'] = df['comorbidities'].str.replace(r'\s*, \s*', ',', regex=True)

    return df


def drop_sparse_columns(df:pd.DataFrame, threshold:float=0.5) -> pd.DataFrame:
    """
    Drop the columns that have more than the threshold missing values,
    except for the protected columns that are crucial for the analysis and should be kept regardless of missingness.
    """
    missing_percent = df.isnull().mean()
    protected_cols = ['das28_score', 'crp', 'esr']
    cols_exceeding_threshold = missing_percent[missing_percent > threshold].index

    cols_to_drop = [col for col in cols_exceeding_threshold if col not in protected_cols]

    if len(cols_to_drop) > 0:
        print(f"Dropping columns with > {threshold*100}% missing values: {','.join(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)
    
    return df



#--------------------------------------------
#                Feature Engineering
#--------------------------------------------



def add_birth_year(df:pd.DataFrame) -> pd.DataFrame:
    """
    extract birth year from date_of_birth
    """
    #idx = int(df.columns.get_loc('date_of_birth'))
    df['birth_year'] = df['date_of_birth'].dt.year.astype('Int64')
    df.drop(columns=['date_of_birth'], inplace=True)
    #df.insert(idx, 'birth_year', years)
    return df


def time_intervals(df:pd.DataFrame) ->pd.DataFrame:

    # ensure chronological order
    df = df.sort_values(['patient_id', 'visit_date'])

    #calculate how many days have passed since the last visit
    #(the very first visit that has not a previous one is filled with 0)
    df['days_since_last_visit'] = (df.groupby('patient_id')['visit_date'].diff().dt.days.fillna(0))

    #calculate disease duration (in years)
    df['disease_duration'] = (df['visit_date'].dt.year - df['year_of_diagnosis']).astype('Int64')

    #calculate age at each visit    
    df['age_at_visit'] = (df['visit_date'].dt.year - df['birth_year'])

    return df


def das28_change(df:pd.DataFrame) -> pd.DataFrame:
    """
    Calculate change in DAS28 score compared to previous visit
    """
    df = df.sort_values(['patient_id', 'visit_date'])
    df['das28_change'] = df.groupby('patient_id')['das28_score'].diff().fillna(0)

    return df


def previous_flare(df:pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature indicating if a flare occurred at the previous visit
    """
    df = df.sort_values(['patient_id', 'visit_date'])
    df['previous_flare'] = df.groupby('patient_id')['flare_event'].shift(1).fillna(0).astype(int)

    return df



def flare_next_visit(df:pd.DataFrame) -> pd.DataFrame:
    """
    Create a target variable indicating if a flare occurs at the next visit
    """
    df = df.sort_values(['patient_id', 'visit_date'])
    df['target_flare_next'] = df.groupby('patient_id')['flare_event'].shift(-1)

    #drop rows where target variable is missing (the last visit of each patient)
    df = df.dropna(subset=['target_flare_next'])
    df['target_flare_next'] = df['target_flare_next'].astype(int)

    return df


def normalization_features(df:pd.DataFrame) -> pd.DataFrame:

    """"
    Normalize continuous features 
    """
    
    df['crp_missing'] = df['crp'].isna().astype(int)
    df['esr_missing'] = df['esr'].isna().astype(int)

    
    df['crp_normalized'] = df['crp']/df['crp_upper_limit']

    crp_high = (df['crp_normalized'] > 1).fillna(False)
    esr_high = (df['esr'] > 20).fillna(False)
    df['inflammation_flag'] = (crp_high | esr_high).astype(int)

    # df['esr_normalized'] = df['esr'] / 20
    # df['inflammation_score'] = (
    # df['crp_normalized'] + df['esr_normalized']
    # ) / 2

    return df

def comorbidities_groups(df:pd.DataFrame) -> pd.DataFrame:
    """"
    divide each comorbidity into groups and create binary features for each group
    count the total comorbidities per patient
    
    """

    comorbidity_mapping = {
        'comorb_cardiovascular': ['transient ischemic attack', 'hypertension', 'coronary artery disease', 'heart failure', 'atrial fibrillation', 'chronic venous insufficiency', 'carotid stenosis', 
                                  'heart attack', 'ventricular septal defect', 'paroxysmal supraventricular tachycardia', 'ascending aortic aneurysm', 'pericarditis'],

        'comorb_metabolic_endocrine': ['hypothyroidism', 'hashimoto', 'dyslipidemia', 'diabetes mellitus', 'cushing syndrome'], 

        'comorb_respiratory' : ['chronic obstructive pulmonary disease', 'asthma', 'pulmonary fibrosis', 'tuberculosis'],

        'comorb_renal_urological' : ['nephrolithiasis', 'prostatitis', 'benign prostatic hyperplasia', 'solitary kidney', 'chronic kidney disease'], 

        'comorb_gastrointestinal_hepatic' : ['hepatitis b', 'resolved hepatitis b', 'gastroesophageal reflux disease', 'total gastrectomy'],

        'comorb_malignancy' : ['kidney cancer', 'breast cancer', 'parotid cancer', 'operated meningioma'], 

        'comorb_musculoskeletal' : ['osteoporosis', 'osteopenia', 'osteoarthritis', 'gout', 'lumbar spondylosis', 'bilateral carpal tunnel syndrome'],

        'comorb_neurologic_psychiatric' : ['depression', 'anxiety disorder', 'parkinsonism', 'vertigo', 'dementia', 'postherpetic neuralgia'],

        'comorb_autoimmune_other' : ['sjogren syndrome', 'thalassemia', 'beta thalassemia minor', 'monoclonal gammopathy of undetermined significance', 'mesenteric lipodystrophy']
                                
                                
    }

 
    for group, diseases in comorbidity_mapping.items():
        pattern = '|'.join(map(re.escape, diseases))
        df[group] = df['comorbidities'].str.contains(pattern, na=False).astype(int)

    df['comorbidity_count'] = df['comorbidities'].apply(lambda x: 0 if x == '' else len(x.split(',')))


    return df

def treatment_decision(df:pd.DataFrame) -> pd.DataFrame:

    
    df['csdmard_use'] = df['csdmard_name'].apply(lambda x: 0 if x == 'None' else 1)
    df['b_ts_dmard_use'] = df['b_ts_dmard_name'].apply(lambda x: 0 if x == 'None' else 1)

    df['steroid_use'] = (df['steroid_dose']>0).astype(int)

    return df


#--------------------------------------------
#                Data Validation
#--------------------------------------------

def validation_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    run sanity checks
    """

    #Checking if there any duplicates and removes them
    duplicates = df.duplicated(subset=['patient_id', 'visit_date']).sum()
    if duplicates > 0:
        print(f"{duplicates} duplicate rows were founded. Removing them...")
        df = df.drop_duplicates(subset=['patient_id', 'visit_date']) 

    #check  if disease duration is negative due to type error
    invalid_duration = df[df['disease_duration']<0]
    if not invalid_duration.empty:
        print('Negative disease duration detected')
        print(invalid_duration[['patient_id', 'visit_date', 'year_of_diagnosis', 'disease_duration']])

    #check if age is unrealistic
    invalid_age = df[(df['age_at_visit']<0) | (df['age_at_visit']>110)]
    if not invalid_age.empty:
        print('Unrealistic age detected.')
        print(invalid_age[['patient_id', 'birth_year', 'age_at_visit', 'visit_date']]) 

    return df

#-----------------------------------------------------
#                Preprocessing Pipeline
#-----------------------------------------------------

def preprocess_data(file_path:str) ->pd.DataFrame:

    df = load_data(file_path)

    #data cleaning
    df = convert_dates(df)  
    df = handle_categorical_missing(df)
    df = encode_features(df)   
    df = comorbidity_clean(df)
    df = drop_sparse_columns(df) 

    #feature engineering
    df = add_birth_year(df)
    df = time_intervals(df)   
    df = das28_change(df)
    
    df = previous_flare(df)
    df = flare_next_visit(df)
    
    df = normalization_features(df)
    df = comorbidities_groups(df)
    df = treatment_decision(df)

    #data validation
    df = validation_data(df)

    return df 

