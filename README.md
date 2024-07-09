Import libraries and define some settings 👇


```python
import sys
sys.path.append("../")
from utils.commons import *
import logging
logging.basicConfig(level=logging.DEBUG) 

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)
```

Create some functions 👇


```python
import logging
def _check_duplicates(df, group_columns):
    logging.debug("1. Checking drop duplicates query")
    if len(df) == len(df.drop_duplicates(subset=group_columns)):
        logging.debug("1. Drop duplicates passed")
    else:
        logging.warning("There are some duplicates.")
        duplicates_example = (df
                              .groupby(group_columns)
                              .agg(counts=(group_columns[0], 'count'))
                              .sort_values(by=['counts'], ascending=False)
                              .head()
                              )
        return duplicates_example
```


```python
df = pd.read_csv("../data/FILE.csv")
```


```python
group_columns = ['Email']
df.drop_duplicates(subset=group_columns).shape
```

Normalize column names 👇


```python
df = _normalize_column_names(df)
```

Cast to date 👇


```python
date_columns = [column for column in df.columns if re.search("date", column)] + ['date_var_with_other_name', 'date_var_with_other_name2']
df = _cast_columns_to_date(df, date_columns, format="%m/%d/%Y")
```

Let's check it out 👇


```python
df_last = _get_last_register(df, ['email'], 'some_date_column')
print(df_last.shape)
```


```python
df['home_office_country'] = [_get_country_from_city(city) for city in df.home_office]
```


```python
group_columns = ['id', 'date']
_check_duplicates(df, group_columns)
```

## EDA


```python
numerical_features = list([col for col in df.select_dtypes(['float64', 'int64']).columns])
categorical_features = list([col for col in df.select_dtypes('object').columns])
```


```python
_numerical_eda(df, numerical_features)
_categorical_eda(df, categorical_features)
```

## JOINS


```python
data = pd.merge(users, demographics_df, on='id', how='left')
data = pd.merge(data, logins_df, on='id', how='left')
data = pd.merge(data, logins_pivot, on='id', how='left')
data = pd.merge(data, trx_df, on='id', how='left')
data = pd.merge(data, trx_pivot, on='id', how='left')
```

## MISSING VALUES


```python
data['DEMO_IngresoSueldos'].fillna(value=0, inplace=True)
na_cat_columns = ['DEMO_EstadoCivil', 'DEMO_NivelEstudios', 'DEMO_SituacionLaboral']
data = df.dropna(subset=na_cat_columns)
```

## PLOTS


```python
def _plt_histogram(df: pd.DataFrame,
                   features: List[str]
                  ):
    """
    Plot a histogram for every numerical column in features argument
    """
    df[features].hist(figsize=(10, 8))
    plt.tight_layout()
    plt.show()
    
def _plt_boxplot(df: pd.DataFrame,
                 features: List[str]
                ):
    """
    Plot a boxplot for every numerical column in features argument
    """
    k = _plt_boxplot_grid(features)
    plt.figure(figsize=(10, k*10))
    
    props_dict = {'linewidth': 2,
                  'color': '#2773B2'
                 }
    
    for n, ticker in enumerate(features):
        ax = plt.subplot(k*3, 3, n + 1)

        df[[ticker]].boxplot(boxprops=props_dict,
                             medianprops=props_dict,
                             whiskerprops=props_dict,
                             capprops=props_dict
                            )
    plt.tight_layout()
    plt.show()

def _plt_boxplot_grid(features):    
    return math.ceil(len(features)/3)
```


```python
_plt_histogram(df, numerical_features)
_plt_boxplot(df, numerical_features)
```

## OUTLIERS


```python
df_out = _label_outliers(df, numerical_features)
_query = "outlier_DEMO_Edad == False and outlier_DEMO_IngresoSueldos == False and outlier_LOGINS_login_count == False and outlier_TRX_monto_sum == False and outlier_TRX_monto_mean == False and outlier_TRX_trx_count == False"
df_no_outliers = df_out.query(_query)
```

## CORR


```python
corr_matrix = df[numerical_features].corr()
corr_matrix
```
