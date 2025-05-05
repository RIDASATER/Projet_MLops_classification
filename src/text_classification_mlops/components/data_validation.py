import pandas as pd

def validate_data(df: pd.DataFrame, schema_path: str) -> bool:
    """Valide le dataframe selon le schéma"""
    import yaml
    
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    
    # Validation des colonnes
    required_cols = [k for k,v in schema['features'].items() if v['required']]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Colonnes manquantes")
    
    # Validation des types
    for col, props in schema['features'].items():
        if col in df.columns:
            if not pd.api.types.is_dtype_equal(df[col].dtype, props['type']):
                raise TypeError(f"Type incorrect pour {col}")
    
    return True