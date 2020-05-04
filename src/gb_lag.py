import pandas as pd 
import numpy as np 
from sales_data import SalesData


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



if __name__ == '__main__':
    model_input = feature_matrix(daily_threshold=500)

