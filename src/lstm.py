import pandas as pd
import numpy as np
from sales_data import SalesData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM



SHOP_GRP_1 = [31, 25, 28, 42, 54, 27, 57, 12, 6]
SHOP_GRP_2 = [18, 50, 47, 46, 35, 24, 58, 15, 26, 7, 38, 19, 21, 43, 22, 56, 16, 29, 53]
SHOP_GRP_3 = [55, 14, 30, 41, 37, 59, 52, 2, 45, 4, 5, 44, 3, 17, 48, 51, 49, 10, 39, 9]
SHOP_GRP_4 = [34, 0, 20, 13, 33, 32, 23, 40, 1, 8]
SHOP_GRP_5 = [11, 36]


class ModelInput(object):

    def __init__(self, sales_data):
        self.data = sales_data
        self.shop_item_df = None

        self.shop_grp_1 = None
        self.shop_grp_2 = None
        self.shop_grp_3 = None
        self.shop_grp_4 = None
        self.shop_grp_5 = None

        self.prev_years_df = None
        self.current_year_df = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def set_shop_groups(self):
        self.shop_grp_1 = self.data.loc[
            self.data['shop_id'].isin([str(x) for x in SHOP_GRP_1])].copy()
        self.shop_grp_2 = self.data.loc[
            self.data['shop_id'].isin([str(x) for x in SHOP_GRP_2])].copy()
        self.shop_grp_3 = self.data.loc[
            self.data['shop_id'].isin([str(x) for x in SHOP_GRP_3])].copy()
        self.shop_grp_4 = self.data.loc[
            self.data['shop_id'].isin([str(x) for x in SHOP_GRP_4])].copy()
        self.shop_grp_5 = self.data.loc[
            self.data['shop_id'].isin([str(x) for x in SHOP_GRP_5])].copy()


    def concat_shop_item(self):
        self.shop_item_df.loc[:, 'shop_item'] = \
            self.shop_item_df['shop_id'].astype(str) + '_' + \
                self.shop_item_df['item_id'].astype(str)


    def create_crosstab(self):
        self.shop_item_df = pd.crosstab(
            index=self.shop_item_df['shop_item'],
            columns=self.shop_item_df['year_month'],
            values=self.shop_item_df['item_cnt_month'],
            aggfunc='sum')
        self.shop_item_df.fillna(0, inplace=True)


    def date_to_cols(self, df):
        """Return sales data aggregated to shop-item with each period
        as a separate feature.
        """

        self.shop_item_df = df.loc[:, [
            'shop_id', 'item_id', 'year_month', 'item_cnt_month']].copy()

        self.concat_shop_item()
        self.create_crosstab()


    def stack_periods(self):
        cols_2013 = []
        cols_2014 = []
        cols_2015 = []

        for col in self.shop_item_df.columns:
            if '2013' in col:
                cols_2013.append(col)
            if '2014' in col:
                cols_2014.append(col)
            if '2015' in col:
                cols_2015.append(col)

        input_2013 = self.shop_item_df.loc[:, cols_2013].copy()
        input_2014 = self.shop_item_df.loc[:, cols_2014].copy()
        input_2015 = self.shop_item_df.loc[:, cols_2015].copy()

        input_2013.drop('201312', axis=1, inplace=True)
        input_2014.drop('201412', axis=1, inplace=True)

        for input_year in [input_2013, input_2014, input_2015]:
            for col in input_year:
                input_year.rename(columns={col: f'P{col[-2:]}'}, inplace=True)

        self.prev_years_df = pd.concat([input_2013, input_2014], axis=0)
        self.current_year_df = input_2015.copy()

    
    def mi_train_test_split(self, test_size=0.20, random_state=123):
        X = self.prev_years_df.iloc[:, :-1].copy()
        y = self.prev_years_df.iloc[:, -1:].copy()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, 
                test_size=test_size, 
                random_state=random_state)


class LSTMModel(object):

    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = Sequential()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def add(self):
        self.model.add(LSTM(
            units=1, 
            batch_input_shape=(
                1, self.X_train.shape[1], self.X_train.shape[2]),
            # return_sequences=True,
            # return_state=True,
            dropout=0.1,
            stateful=True))

        self.model.add(Dense(1))
        self.model.add(Dense(1))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def fit(self, epochs=30):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=1, 
            verbose=1, 
            shuffle=False)


    def predict(self):
        self.model.predict(self.X_test, batch_size=1)


if __name__ == '__main__':
    shop_item_df = shop_item_df(daily_threshold=500)

