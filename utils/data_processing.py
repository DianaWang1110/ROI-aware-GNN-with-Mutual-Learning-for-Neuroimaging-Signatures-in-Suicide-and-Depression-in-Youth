import pandas as pd

def make_column_names_unique(df):
    """Ensure column names are unique by appending suffixes where needed."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        dup_indices = cols[cols == dup].index.tolist()
        cols[dup_indices] = [f'{dup}_{i+1}' if i != 0 else dup for i in range(len(dup_indices))]
    df.columns = cols
    return df

def drop_duplicate_columns(df, column_name='src_subject_id'):
    """Remove duplicate columns with the same name."""
    if df.columns.duplicated().sum() > 0:
        df = df.loc[:, ~df.columns.duplicated()]
    return df
