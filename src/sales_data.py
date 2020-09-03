import pandas as pd
import numpy as np
from datetime import datetime
import nlp as nlp



class SalesData(object):

    def __init__(self):
        self.daily_sales = None
        self.monthly_sales = None
        self.shops = None
        self.item_categories = None
        self.items = None
        self.lag_df = None
        self.feature_matrix = None


    def daily_drop_outliers(self, daily_threshold):
        """Drop outliers by filtering out data where
        item_cnt_dat < daily_threshold. Determined through previous EDA.

        :param daily_threshold: max item count
        :type daily_threshold: int
        """
        self.daily_sales = self.daily_sales.loc[
            self.daily_sales['item_cnt_day'] < daily_threshold]


    def fetch_shop_sales(self, daily_threshold):
        """Fetch sales data."""
        self.daily_sales = pd.read_csv('../data/sales_train.csv')
        self.daily_drop_outliers(daily_threshold)


    def format_sales(self):
        """Format shop_id, item_id, and date columns."""
        df = self.daily_sales.copy()

        for col in ['shop_id', 'item_id']:
            df[col] = df[col].astype(str)

        df['date'] = df['date'].apply(
            lambda x: datetime.strptime(x, '%d.%m.%Y'))

        self.daily_sales = df


    def break_out_period_data(self):
        """Break out month and year, and create year_month column
        from date."""
        df = self.daily_sales.copy()

        df['year'] = df['date'].apply(lambda x: x.year)
        df['month'] = df['date'].apply(lambda x: f'{x.month:02d}')
        df['year_month'] = df['year'].astype(str) + df['month']

        self.daily_sales = df


    def monthly_agg(self):
        """Get total item counts and average item price by month,
        shop, item."""
        print('Aggregating for monthly sales...')
        gb_cols = ['date_block_num', 'year', 'month', 'year_month',
            'shop_id', 'item_id']

        monthly_counts = \
            self.daily_sales.groupby(gb_cols)['item_cnt_day'] \
                .sum().reset_index()
        monthly_avg_price = \
            self.daily_sales.groupby(gb_cols)['item_price'] \
                .mean().reset_index()

        self.monthly_sales = pd.merge(
            monthly_counts, monthly_avg_price, on=gb_cols, how='inner')

        self.monthly_sales.rename(
            columns={'item_cnt_day': 'item_cnt_month',
                     'item_price': 'avg_price'}, inplace=True)


    def calc_daily_sales(self):
        """Add column for total daily sales by shop/item."""
        item_daily_sales = np.array(
            self.daily_sales.loc[:, ['item_price', 'item_cnt_day']])

        self.daily_sales['sales_day'] = np.prod(item_daily_sales, axis=1)


    def calc_monthly_sales(self):
        """Add column for total monthly sales."""
        if self.monthly_sales is None:
            self.monthly_agg()

        item_ttl_sales = np.array(
            self.monthly_sales.loc[:, ['avg_price', 'item_cnt_month']])

        self.monthly_sales['sales_month'] = np.prod(item_ttl_sales, axis=1)


    def set_sales_data(self, daily_threshold=500):
        """Set data and apply formatting."""
        self.fetch_shop_sales(daily_threshold)

        print('Formatting and aggregating sales data...')
        self.format_sales()
        self.break_out_period_data()
        self.monthly_agg()
        self.calc_daily_sales()
        self.calc_monthly_sales()
        self.calc_daily_sales()
        self.calc_monthly_sales()


    def fetch_shop_data(self):
        """Fetch shop data."""
        self.shops = pd.read_csv('../data/shops.csv')
        self.shops['shop_id'] = self.shops['shop_id'].astype(str)


    def drop_redundant_shop_cols(self):
        """Drop redundant shop data."""
        drop_cols = ['shop_name', 'clean_name']
        for col in drop_cols:
            self.shops.drop(col, axis=1, inplace=True)


    def shop_cat_info(self):
        """Add columns for shop categories (i.e. MEGA, mall) and location name
        (i.e. city) to self.shops using NLP."""

        self.shops['clean_name'] = nlp.clean_names(self.shops['shop_name'])
        self.shops['loc_name'] = self.shops['clean_name'].apply(
            lambda x: x.split()[0])
        self.shops = nlp.get_top_words(
            self.shops, self.shops['clean_name'], 10)

        # remove top occurences if in loc_name
        for col in self.shops.columns:
            if col.upper() in self.shops['loc_name'].unique():
                self.shops.drop(col, axis=1, inplace=True)


    def merge_shops_to_daily(self):
        """Merges shops to sales."""
        self.daily_sales = pd.merge(
            self.daily_sales, self.shops, on='shop_id', how='inner')


    def merge_shops_to_monthly(self):
        """Merges shops to sales."""
        self.monthly_sales = pd.merge(
            self.monthly_sales, self.shops, on='shop_id', how='inner')


    def merge_shop_data_to_sales(self):
        """Fetch shop data and merge to sales data."""
        self.fetch_shop_data()
        self.shop_cat_info()
        self.drop_redundant_shop_cols()
        self.merge_shops_to_daily()
        self.merge_shops_to_monthly()


    def fetch_item_data(self):
        """Fetch item category data from csv files."""
        self.item_categories = pd.read_csv('../data/item_categories.csv')
        itms = pd.read_csv('../data/items.csv')

        self.items = pd.merge(
            itms, self.item_categories, on='item_category_id', how='inner')
        self.items['item_id'] = self.items['item_id'].astype(str)


    def merge_items_to_daily(self):
        """Add item info to sales."""
        self.daily_sales = pd.merge(
            self.daily_sales, self.items, on='item_id', how='inner')


    def merge_items_to_monthly(self):
        """Add item info to sales."""
        self.monthly_sales = pd.merge(
            self.monthly_sales, self.items, on='item_id', how='inner')


    def merge_item_data_to_sales(self):
        """Fetch and merge categorical item data to daily and monthly
        sales."""

        self.fetch_item_data()
        self.merge_items_to_daily()
        self.merge_items_to_monthly()


    def add_missing_dates(self, df):
        """[summary]

        :param df: [description]
        :type df: [type]
        :return: [description]
        :rtype: [type]
        """
        date_idx = pd.date_range(df['date'].min(), df['date'].max())
        date_df = pd.DataFrame(data=date_idx, columns=['date'])
        df = pd.merge(date_df, df, on='date', how='left')
        mask = df['item_cnt_day'].isna()
        df.loc[mask, 'item_cnt_day'] = 0
        df.fillna(method='ffill', inplace=True)
        return df


    def add_missing_periods(self, df):
        maxdate = datetime.strptime('20151001', '%Y%m%d')
        date_idx = pd.date_range(
            df.index.min(), maxdate, freq='MS')
        date_df = pd.DataFrame(data=date_idx, columns=['plot_date'])
        df = pd.merge(date_df, df, on='plot_date', how='left')
        mask = df['item_cnt_month'].isna()
        df.loc[mask, 'item_cnt_month'] = 0
        df.set_index(keys='plot_date', inplace=True)
        return df


    def set_lag(self):
        model_input = lag.feature_matrix(daily_threshold=500)
        self.lag_df = lag.stack_lag_dfs(model_input)


if __name__ == '__main__':
    sd = SalesData()

    sd.set_sales_data()
    sd.merge_shop_data_to_sales()
    sd.merge_item_data_to_sales()