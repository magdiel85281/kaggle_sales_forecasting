import pandas as pd
import os
import re
from datetime import datetime


class SalesData(object):

    def __init__(self):
        self.daily_sales = None
        self.monthly_sales = None
        self.shops = None
        self.item_categories = None
        self.items = None

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

    def set(self):
        """Set data."""
        self.shops = pd.read_csv('data/shops.csv')
        self.item_categories = pd.read_csv('data/item_categories.csv')
        self.items = pd.read_csv('data/items.csv')
        self.daily_sales = pd.read_csv('data/sales_train.csv')

        print('Formatting sales data...')
        self.format_sales()
        self.break_out_period_data()

    def monthly_agg(self):
        """Get sum of item_cnt_day by month, shop, item."""

        gb_cols = ['date_block_num', 'year', 'month', 'year_month',
                   'shop_id', 'item_id', 'item_price']
        self.monthly_sales = \
            self.daily_sales.groupby(gb_cols)['item_cnt_day'] \
                .sum().reset_index()
        self.monthly_sales.rename(
            columns={'item_cnt_day': 'item_cnt_mth'}, inplace=True)