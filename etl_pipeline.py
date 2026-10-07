import pandas as pd
from sqlalchemy import create_engine

class CardioETLPipeline:
    """
    Extracts raw cardiovascular data, transforms it by cleaning and filtering outliers,
    and loads the cleaned dataset into a PostgreSQL database.
    """
    def __init__(self, data_path: str, db_url: str):
        self.data_path = data_path
        self.db_url = db_url

    def extract(self) -> pd.DataFrame:
        print(f"Extracting data from {self.data_path}...")
        return pd.read_csv(self.data_path, sep=';')

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Transforming and cleaning data...")
        initial_count = len(df)
        
        # 1. Feature Engineering: Convert age from days to years
        df['age'] = (df['age'] / 365.25).round().astype(int)
        
        # 2. Outlier Removal: Strict Data Cleaning (Biological Impossibilities)
        bp_condition = (
            (df['ap_hi'] >= 60) & (df['ap_hi'] <= 250) &
            (df['ap_lo'] >= 40) & (df['ap_lo'] <= 180) &
            (df['ap_hi'] > df['ap_lo'])
        )
        df = df[bp_condition].copy()
        
        # 3. Clinical Feature Engineering
        df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
        df['map'] = ((df['ap_hi'] + 2 * df['ap_lo']) / 3).round(2)
        
        # Drop the 'id' column as it has no predictive value
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            
        print(f"Transformation complete. Removed {initial_count - len(df)} outlier/invalid records.")
        return df

    def load(self, df: pd.DataFrame, table_name: str = 'cleaned_cardio_data'):
        print(f"Loading data into PostgreSQL database (Table: {table_name})...")
        engine = create_engine(self.db_url)
        # Load the cleaned data into the database
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print("Data successfully loaded into PostgreSQL.")

    def run(self):
        df_raw = self.extract()
        df_clean = self.transform(df_raw)
        self.load(df_clean)

if __name__ == "__main__":
    # ⚠️ IMPORTANT: Update these credentials to match your local PostgreSQL setup
    # Format: postgresql://username:password@localhost:5432/database_name
    DB_URL = 'postgresql+psycopg2://postgres:hybesty123@localhost:5432/cardio_db'
    
    pipeline = CardioETLPipeline(data_path='cardio_train.csv', db_url=DB_URL)
    pipeline.run()
