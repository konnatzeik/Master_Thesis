import pandas as pd

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
    df = pd.merge(df_static, df_visits, on='patient_id', how='right')

    return df



#--------------------------------------------
#                Data Cleaning
#--------------------------------------------

def check_for_duplicates(df:pd.DataFrame) -> pd.DataFrame:
    """
    Checking if there any duplicates and removes them
    """
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows. Removing them...")
        df = df.drop_duplicates()
    
    return df


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
        df['switch_reason'] = df['switch_reason'].fillna('No_Switch')

    if 'csdmard_name' in df.columns:
        df['csdmard_name'] = df['csdmard_name'].fillna('None')

    if 'b_ts_dmard_name' in df.columns:
        df['b_ts_dmard_name'] = df['b_ts_dmard_name'].fillna('None')

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

    #calculate how many days have passed since the last visit (the very first visit that has not a previous one is filled with 0)
    df['days_since_last_visit'] = (df.groupby('patient_id')['visit_date'].diff().dt.days.fillna(0))

    #calculate disease duration (in years)
    df['disease_duration'] = (df['visit_date'].dt.year - df['year_of_diagnosis']).astype('Int64')

    return df


def age_calculation(df:pd.DataFrame) -> pd.DataFrame:
    """
    Calculate age at each visit
    """
    df['age_at_visit'] = (df['visit_date'].dt.year - df['birth_year'])

    return df


def das28_change(df:pd.DataFrame) -> pd.DataFrame:
    """
    Calculate change in DAS28 score compared to previous visit
    """
    df = df.sort_values(['patient_id', 'visit_date'])
    df['das28_change'] = df.groupby('patient_id')['das28_score'].diff().fillna(0)

    return df

def flare_next_visit(df:pd.DataFrame) -> pd.DataFrame:
    """
    Create a target variable indicating if a flare occurs at the next visit
    """
    df = df.sort_values(['patient_id', 'visit_date'])
    df['target_flare_next'] = df.groupby('patient_id')['flare_event'].shift(-1)

    return df



#--------------------------------------------
#                Data Validation
#--------------------------------------------

def validation_data(df:pd.DataFrame) -> None:
    """
    run sanity checks
    """

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

    

#-----------------------------------------------------
#                Preprocessing Pipeline
#-----------------------------------------------------

def preprocess_data(file_path:str) ->pd.DataFrame:

    df = load_data(file_path)
    df = check_for_duplicates(df)
    df = convert_dates(df)
    df = add_birth_year(df)
    df = time_intervals(df)
    df = age_calculation(df)
    df = handle_categorical_missing(df)
    df = das28_change(df)
    df = flare_next_visit(df)


    validation_data(df)


    return df 


#--------------------------------------------
#                Execution
#--------------------------------------------

if __name__ == "__main__":

    file_path = r"data/RA_Dataset.xlsx"
    output_path = r"data/RA_dataset_cleaned.csv"

    df_cleaned = preprocess_data(file_path)

    # Save final processed dataset to CSV
    df_cleaned.to_csv(output_path, index=False)
