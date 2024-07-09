from typing import List
import pandas as pd
import re

__all__ = ['List', 'pd', 're',
           #
           '_normalize_column_names',
           '_get_last_register',
           '_cast_columns_to_date',
           '_get_country_from_city',
           #
           '_numerical_eda',
           '_categorical_eda',
           '_add_prefix_in_columns',
           #
           '_label_outliers',
           
          ]

# OUTLIERS ############################################################################################################
def _label_outliers(df: pd.DataFrame,
                    numerical_features: List[str]
                   ) -> pd.DataFrame:
    """
    Perform tukey method for outliers in every numerical_feature
    """
    for feature in numerical_features:
        df_outliers = _label_outliers_tukey(df, feature, f'outlier_{feature}')
    return df_outliers

def _label_outliers_tukey(df: pd.DataFrame,
                          column: str,
                          outlier_column: str
                         ) -> pd.DataFrame:
    """
    Perform tukey method to identify outliers
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    df[outlier_column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), True, False)
    return df

# EDA ############################################################################################################
def _numerical_eda(df: pd.DataFrame,
                   features: List[str]
                  ) -> pd.DataFrame:
    """
    Exploratory data analysis of numerical features
    """
    df = df[features]

    len_list = []
    null_list = []
    prop_null_list = []
    mean_list = []
    std_list = []
    min_list = []
    per_1_list = []
    per_25_list = []
    median_list = []
    per_75_list = []
    per_99_list = []
    max_list = []

    for i in features:
        len_v = len(df)
        len_list.append(len_v)
        null_v = df[i].isnull().sum()
        null_list.append(null_v)
        prop_null_v = null_v/len_v
        prop_null_list.append(prop_null_v)
        mean_v = np.mean(df[i])
        mean_list.append(mean_v)
        std_v = np.std(df[i])
        std_list.append(std_v)
        min_v = np.min(df[i])
        min_list.append(min_v)
        per_1_v = np.percentile(df[i].dropna(), 1)
        per_1_list.append(per_1_v)
        per_25_v = np.percentile(df[i].dropna(), 25)
        per_25_list.append(per_25_v)
        median_v = np.median(df[i].dropna())
        median_list.append(median_v)
        per_75_v = np.percentile(df[i].dropna(), 75)
        per_75_list.append(per_75_v)
        per_99_v = np.percentile(df[i].dropna(), 99)
        per_99_list.append(per_99_v)
        max_v = np.max(df[i])
        max_list.append(max_v)

    df = pd.DataFrame({"feature": features,
                       "n_row": len_list,
                       "n_col": len(features),
                       "num_null": null_list,
                       "prop_null": prop_null_list,
                       "mean": mean_list,
                       "std": std_list,
                       "min": min_list,
                       "per_1": per_1_list,
                       "per_25": per_25_list,
                       "median": median_list,
                       "per_75": per_75_list,
                       "per_99": per_99_list,
                       "max": max_list
                      })    
    df  = df.sort_values(by=["feature"])
    df = df.reset_index(drop=True)
    return df


def _categorical_eda(df: pd.DataFrame,
                     features: List[str]
                    ) -> pd.DataFrame:
    """
    Exploratory data analysis of categorical features
    """
    df = df[features]

    len_list = []
    null_list = []
    prop_null_list = []
    category_list = []
    num_category_list = []
    mode_list = []
    mode_count_list = []

    for i in features:
        len_v = len(df)
        len_list.append(len_v)
        null_v = df[i].isnull().sum()
        null_list.append(null_v)
        prop_null_v = null_v/len_v
        prop_null_list.append(prop_null_v)
        category_v = df[i].unique()
        category_list.append(category_v)
        num_category_v = len(category_v)
        num_category_list.append(num_category_v)
        mode_v = df[i].mode()[0]
        mode_list.append(mode_v)
        mode_count_v = len(df[i][df[i] == mode_v])
        mode_count_list.append(mode_count_v)

    df = pd.DataFrame({"feature": features,
                       "n_row": len_list,
                       "n_col": len(features),
                       "num_null": null_list,
                       "prop_null": prop_null_list,
                       "num_categories": num_category_list,
                       "category": category_list,
                       "mode": mode_list,
                       "mode_count": mode_count_list,
                      })
    df  = df.sort_values(by=["feature"])
    df = df.reset_index(drop=True)
    return df

def _add_prefix_in_columns(df, prefix, exclude):
    column_names = []
    for col in df.columns:
        if col in exclude:
            column_names.append(col)
        else:
            column_names.append(prefix + str(col))
    return column_names

# PREPROCESSING ####################################################################################################
def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the column names of a Pandas DataFrame.
    
    Parameters:
        df (DataFrame): The Pandas DataFrame whose column names need to be normalized.
    
    Returns:
        DataFrame: The DataFrame with normalized column names.
    """
    df.columns = df.columns.str.lower()
    
    df.columns = df.columns.str.replace('\n', '_')
    df.columns = df.columns.str.replace(' ', '_')
    #df.columns = df.columns.str.replace('[^\w\s]', '')
    return df

def _get_last_register(df: pd.DataFrame,
                       window_partition_by: List[str],
                       order_by: str,
                       last_register: int = 1
                      ) -> pd.DataFrame:
    """
    Create DataFrame partitioned by columns and ordered by a specified column.
    
    Args:
        df (DataFrame): Input DataFrame.
        window_partition_by (List[str]): List of column names to partition by.
        order_by (str): Column name to order by.
        last_register (int): Number of last records to return.

    Returns:
        DataFrame: DataFrame containing the last N records based on the partition and order criteria.
    """
    df_sorted = df.sort_values(by=order_by, ascending=False)
    last_df = df_sorted.groupby(window_partition_by).head(last_register)
    return last_df

def _cast_columns_to_date(df: pd.DataFrame,
                          date_columns: List[str],
                          format: str
                          ) -> pd.DataFrame:
    """
    Cast specified columns of a DataFrame to date format.

    Args:
        df (DataFrame): Input DataFrame.
        date_columns (List[str]): List of column names to cast to date format.

    Returns:
        DataFrame: DataFrame with specified columns cast to date format.
    """
    return df.assign(**{col: pd.to_datetime(df[col], format=format, errors='coerce') for col in date_columns})

def _get_country_from_city(city: str) -> str:
    """
    Get the name of the country based on a given city.

    Args:
        city (str): Name of the city.

    Returns:
        str: Name of the country (Ecuador, Chile, Brazil), or "Unknown" if the city is not from any of these countries.
    """
    city = city.lower()
    
    ecuador_cities = [
        "ambato",
        "azogues",
        "cuenca",
        "durán",
        "esmeraldas",
        "guaranda",
        "guayaquil",
        "ibarra",
        "latacunga",
        "loja",
        "machala",
        "manta",
        "milagro",
        "portoviejo",
        "quito",
        "riobamba",
        "santa elena",
        "santo domingo",
        "tulcán"
    ]
    
    brazil_cities = [
        "são paulo",
        "sao paulo",
        "rio de janeiro",
        "salvador",
        "brasília",
        "fortaleza",
        "curitiba",
        "belo horizonte",
        "manaus",
        "recife",
        "belém",
        "porto alegre",
        "goiânia",
        "guarulhos",
        "campinas",
        "natal",
        "niteroi",
        "são luís",
        "cuiabá",
        "florianópolis",
        "maceió",
        "teresina",
        "joão pessoa",
        "londrina",
        "aracaju",
        "campo grande",
        "santos",
        "vila velha",
        "contagem",
        "sobral",
        "osasco",
        "são bernardo do campo"
    ]
    
    chile_cities = [
        "santiago",
        "valparaíso",
        "viña del mar",
        "la serena",
        "antofagasta",
        "iquique",
        "concepción",
        "arica",
        "punta arenas",
        "temuco",
        "rancagua",
        "talcahuano",
        "puerto montt",
        "calama",
        "ovalle",
        "coquimbo",
        "chillán",
        "puerto varas",
        "valdivia",
        "talca",
        "arauco",
        "castro",
        "coyhaique",
        "coronel",
        "angol",
        "curicó",
        "osorno",
        "nacimiento",
        "molina",
        "villarrica",
        "san antonio",
        "los angeles",
        "san felipe",
        "pucón",
    ]

    if city in ecuador_cities:
        return "Ecuador"
    elif city in chile_cities:
        return "Chile"
    elif city in brazil_cities:
        return "Brazil"
    else:
        return "Unknown"