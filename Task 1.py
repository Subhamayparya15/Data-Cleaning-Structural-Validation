import pandas as pd
import numpy as np
import logging

# Configure logging for pipeline traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaningPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.metrics = {"initial_rows": len(df), "initial_cols": len(df.columns)}
        
    def profile_data(self) -> 'DataCleaningPipeline':
        """Step 1: Exploratory Data Profiling"""
        logging.info("--- 1. Exploratory Data Profiling ---")
        logging.info(f"Shape: {self.df.shape}")
        
        missing_counts = self.df.isnull().sum()
        missing_ratios = (missing_counts / len(self.df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Missing Percentage (%)': missing_ratios})
        
        logging.info("\nColumns with missing values:\n" + str(missing_df[missing_df['Missing Count'] > 0]))
        return self

    def clean_duplicates_and_strings(self, string_cols: list = None) -> 'DataCleaningPipeline':
        """Step 2: Deduplication & String Standardization"""
        logging.info("--- 2. Deduplication & String Standardization ---")
        
        # Remove exact duplicates
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        dropped_duplicates = initial_rows - len(self.df)
        self.metrics["duplicates_dropped"] = dropped_duplicates
        logging.info(f"Dropped {dropped_duplicates} exact duplicate rows.")
        
        # Standardize string columns (strip whitespace, lowercase, fix multi-spaces)
        if string_cols:
            for col in string_cols:
                if col in self.df.columns:
                    self.df[col] = (
                        self.df[col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .replace(r'\s+', ' ', regex=True)
                    )
                    # Convert 'nan' string artifacts back to proper NaN
                    self.df[col] = self.df[col].replace('nan', np.nan)
            logging.info(f"Standardized text formatting for columns: {string_cols}")
            
        return self

    def handle_missing_values(self, num_strategy: str = 'median', drop_threshold: float = 0.4) -> 'DataCleaningPipeline':
        """Step 3: Strategic Missing Value Treatment"""
        logging.info("--- 3. Strategic Missing Value Treatment ---")
        
        # Drop columns exceeding the missingness threshold
        null_ratios = self.df.isnull().mean()
        cols_to_drop = null_ratios[null_ratios > drop_threshold].index.tolist()
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            logging.info(f"Dropped columns due to high missingness (> {drop_threshold*100}%): {cols_to_drop}")
        self.metrics["columns_dropped_high_missing"] = cols_to_drop
        
        # Impute remaining numerical columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.df[col].isnull().sum() > 0:
                fill_value = self.df[col].median() if num_strategy == 'median' else self.df[col].mean()
                self.df[col].fillna(fill_value, inplace=True)
                logging.info(f"Imputed numerical column '{col}' with {num_strategy}: {fill_value}")
                
        # Impute remaining categorical/object columns with mode
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                mode_vals = self.df[col].mode()
                if not mode_vals.empty:
                    fill_value = mode_vals[0]
                    self.df[col].fillna(fill_value, inplace=True)
                    logging.info(f"Imputed categorical column '{col}' with mode: {fill_value}")
                    
        return self

    def normalize_formats(self, date_cols: list = None, numeric_cleaning_cols: list = None) -> 'DataCleaningPipeline':
        """Step 4: Format Normalization & Type Coercion"""
        logging.info("--- 4. Format Normalization & Type Coercion ---")
        
        # Standardize dates to datetime objects
        if date_cols:
            for col in date_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    logging.info(f"Converted column '{col}' to datetime format.")
                    
        # Clean currency/numeric strings (remove symbols like $, commas)
        if numeric_cleaning_cols:
            for col in numeric_cleaning_cols:
                if col in self.df.columns and self.df[col].dtype == 'object':
                    self.df[col] = (
                        self.df[col]
                        .astype(str)
                        .str.replace(r'[$,]', '', regex=True)
                        .str.strip()
                    )
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    logging.info(f"Cleaned and casted numeric string column '{col}'.")
                    
        return self

    def validate_schema(self, primary_key: str = None) -> pd.DataFrame:
        """Step 5: Structural Validation & Integrity Assertions"""
        logging.info("--- 5. Structural Validation & Integrity Assertions ---")
        
        # Check for lingering null values
        total_nulls = self.df.isnull().sum().sum()
        if total_nulls > 0:
            logging.warning(f"Validation Warning: {total_nulls} null values remain in the dataset.")
        else:
            logging.info("Validation Check Passed: Zero missing values remaining.")
            
        # Check primary key uniqueness if specified
        if primary_key and primary_key in self.df.columns:
            duplicates = self.df[primary_key].duplicated().sum()
            assert duplicates == 0, f"Primary key violation: Found {duplicates} duplicate keys in '{primary_key}'."
            logging.info(f"Validation Check Passed: Primary key '{primary_key}' is unique.")
            
        logging.info(f"Pipeline complete. Final Dataset Shape: {self.df.shape}")
        return self.df


# --- Example Execution ---
if __name__ == "__main__":
    # Create sample dirty dataframe for testing
    raw_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'customer_name': [' Alice Smith ', 'Bob Jones', 'bob jones', 'Charlie Brown', None, '  Diana Prince'],
        'age': [25, 30, 30, np.nan, 45, 200], # 200 is an outlier
        'signup_date': ['2023-01-15', '2023/02/20', '2023/02/20', 'InvalidDate', '2023-04-10', '2023-05-01'],
        'purchase_amount': ['$1,200.50', '$450.00', '$450.00', None, '$3,100.00', '950.00'],
        'unimportant_junk_col': [None, None, None, None, None, None] # Should get dropped (>40% missing)
    }
    
    df_raw = pd.DataFrame(raw_data)
    
    # Run pipeline
    pipeline = DataCleaningPipeline(df_raw)
    cleaned_df = (
        pipeline
        .profile_data()
        .clean_duplicates_and_strings(string_cols=['customer_name'])
        .handle_missing_values(num_strategy='median', drop_threshold=0.4)
        .normalize_formats(date_cols=['signup_date'], numeric_cleaning_cols=['purchase_amount'])
        .validate_schema(primary_key=None) # Set primary_key='id' if testing unique constraint
    )
    
    print("\nCleaned DataFrame Preview:")
    print(cleaned_df.head())
          
