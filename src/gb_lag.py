import pandas as pd 
import numpy as np 
from src.sales_data import SalesData


def add_shop_item(df):
    df.loc[:, 'shop_item'] = \
        df['shop_id'].astype(str) + '_' + df['item_id'].astype(str)
    return df


def create_crosstab(df):
    mdl_input = pd.crosstab(index=df['shop_item'], 
                            columns=df['year_month'], 
                            values=df['item_cnt_month'], 
                            aggfunc='sum')
    return mdl_input.fillna(0)


def feature_matrix(daily_threshold=500):
    """Return sales data aggregated to shop-item with each period
    as a separate feature.
    
    :param daily_threshold: max daily item count threshold, 
        defaults to 500
    :type daily_threshold: int, optional
    :return: dataframe as described above
    :rtype: pd.DataFrame
    """    
    sd = SalesData()
    sd.set(daily_threshold=daily_threshold)
    sales = sd.monthly_sales

    mdl_input = sales.loc[:, [
        'shop_id', 'item_id', 'year_month', 'item_cnt_month']].copy()

    mdl_input = add_shop_item(mdl_input)
    mdl_input = create_crosstab(mdl_input)
    return mdl_input


def stack_lag_dfs(df):
    cols_2013 = []
    cols_2014 = []
    cols_2015 = []

    for col in df.columns:
        if '2013' in col:
            cols_2013.append(col)
        if '2014' in col:
            cols_2014.append(col)
        if '2015' in col:
            cols_2015.append(col)

    input_2013 = df.loc[:, cols_2013].copy()
    input_2014 = df.loc[:, cols_2014].copy()
    input_2015 = df.loc[:, cols_2015].copy()

    input_2013.drop('201312', axis=1, inplace=True)
    input_2014.drop('201412', axis=1, inplace=True)

    for input_year in [input_2013, input_2014, input_2015]:
        for col in input_year:
            input_year.rename(columns={col: f'P{col[-2:]}'}, inplace=True)

    train_df = pd.concat([input_2013, input_2014], axis=0)
    test_df = input_2015.copy()

    return train_df, test_df
    

if __name__ == '__main__':
    model_input = feature_matrix(daily_threshold=500)

