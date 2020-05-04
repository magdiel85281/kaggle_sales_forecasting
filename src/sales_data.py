import pandas as pd
from datetime import datetime
import nlp as nlp


class SalesData(object):

    def __init__(self):
        self.daily_sales = None
        self.monthly_sales = None
        self.shops = None
        self.item_categories = None
        self.items = None

    def daily_drop_outliers(self, daily_threshold):
        self.daily_sales = self.daily_sales.loc[
            self.daily_sales['item_cnt_day'] < daily_threshold]

    def format_sales(self):
        """Format columns."""
        df = self.daily_sales.copy()
        for col in ['shop_id', 'item_id']:
            df[col] = df[col].astype(str)
        df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y'))
        self.daily_sales = df

    def break_out_period_data(self):
        """Break out month and year, and create year_month column from date."""
        df = self.daily_sales.copy()
        df['year'] = df['date'].apply(lambda x: x.year)
        df['month'] = df['date'].apply(lambda x: f'{x.month:02d}')
        df['year_month'] = df['year'].astype(str) + df['month']
        self.daily_sales = df

    def monthly_agg(self):
        """Get sum of item_cnt_day by month, shop, item."""
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

    def set(self, daily_threshold=500):
        """Set data."""
        self.shops = pd.read_csv('data/shops.csv')
        self.daily_sales = pd.read_csv('data/sales_train.csv')
        self.daily_drop_outliers(daily_threshold)
        self.item_categories = pd.read_csv('data/item_categories.csv')
        itms = pd.read_csv('data/items.csv')
        self.items = pd.merge(
            itms, self.item_categories, on='item_category_id', how='inner')

        print('Formatting sales data...')
        self.format_sales()
        self.break_out_period_data()
        self.monthly_agg()

    def calc_sales(self):
        """Add column for daily and monthly sales to respective dataframes."""
        if self.monthly_sales is None:
            self.monthly_agg()
        self.monthly_sales['sales_month'] = \
            self.monthly_sales['avg_price'] * self.monthly_sales['item_cnt_month']
        self.daily_sales['sales_day'] = \
            self.daily_sales['item_price'] * self.daily_sales['item_cnt_day']

    def shop_cat_info(self):
        """Add columns for shop categories (i.e. MEGA, mall) and location name
        (i.e. city) to self.shops."""
        self.shops['clean_name'] = nlp.clean_names(self.shops['shop_name'])
        self.shops['loc_name'] = self.shops['clean_name'].apply(
            lambda x: x.split()[0])
        self.shops = nlp.get_top_words(self.shops, self.shops['clean_name'], 10)

        # remove top occurences if in loc_name
        for col in self.shops.columns:
            if col.upper() in self.shops['loc_name'].unique():
                self.shops.drop(col, axis=1, inplace=True)

    def add_shop_cat_info(self):
        """Merges shops (with categorical info) to sales."""
        self.shop_cat_info()
        self.shops['shop_id'] = self.shops['shop_id'].astype(str)
        self.daily_sales = pd.merge(
            self.daily_sales, self.shops, on='shop_id', how='inner')
        self.monthly_sales = pd.merge(
            self.monthly_sales, self.shops, on='shop_id', how='inner')
        drop_cols = ['shop_name', 'clean_name']
        for df in [self.daily_sales, self.monthly_sales]:
            for col in drop_cols:
                df.drop(col, axis=1, inplace=True)

    def add_item_info(self):
        """Add item info to sales."""
        self.items['item_id'] = self.items['item_id'].astype(str)
        self.daily_sales = pd.merge(
            self.daily_sales, self.items, on='item_id', how='inner')
        self.monthly_sales = pd.merge(
            self.monthly_sales, self.items, on='item_id', how='inner')

    def add_missing_dates(self, df):
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
