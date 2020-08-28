import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout



# assigns shop groups by shop_id after previous EDA on item counts
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
        self.X_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()


    def set_shop_groups(self):
        """Break up shops into groups. Groups determined by sales volume in
        initial EDA.
        """

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
        """Create column shop_item as '_'-separated shop_id and item_id."""
        self.shop_item_df.loc[:, 'shop_item'] = \
            self.shop_item_df['shop_id'].astype(str) + '_' + \
                self.shop_item_df['item_id'].astype(str)


    def create_crosstab(self):
        """Create crosstab from dataset with shop_item in the index,
        year_month as the columns, and item_cnt_mnth as the values."""

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
        """Break up dataset by year and stack (concat), aligning periods.
        Set prev_years_df and current_year_df."""

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
                input_year.rename(
                    columns={col: f'P{col[-2:]}'}, inplace=True)

        self.prev_years_df = pd.concat([input_2013, input_2014], axis=0)
        self.current_year_df = input_2015.copy()


    def train_test_split(self, test_size=0.20, random_state=123):
        """Split the previous years data into a training set and a
        test set.

        :param test_size: proportion of data to allocate to the test
            set, defaults to 0.20
        :type test_size: float, optional
        :param random_state: random state, defaults to 123
        :type random_state: int, optional
        """

        X = self.prev_years_df.iloc[:, :-1].copy()
        y = self.prev_years_df.iloc[:, -1:].copy()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y,
                test_size=test_size,
                random_state=random_state)


    def scale_min_max(self):
        """Scale data using MinMaxScaler."""
        self.X_train = self.X_scaler.fit_transform(self.X_train)
        self.X_test = self.X_scaler.transform(self.X_test)

        self.y_train = self.y_scaler.fit_transform(self.y_train)


    def reshape(self):
        """Reshape data as a 3D array for input into LSTM model."""
        self.X_train = np.reshape(
            self.X_train,
            (self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = np.reshape(
            self.X_test,
            (self.X_test.shape[0], 1, self.X_test.shape[1]))


    def test_results(self, y_pred):
        """Print the root mean squared error by comparing y_pred to
        self.y_test.

        :param y_pred: array of predicted values.
        :type y_pred: np.array
        """
        y_pred_unscaled = self.y_scaler.inverse_transform(y_pred)
        print(f'RMSE: {np.sqrt(mean_squared_error(self.y_test, y_pred_unscaled))}')



class LSTMModel(object):

    def __init__(self, X_train, y_train):
        self.model = Sequential()
        self.X_train = X_train
        self.y_train = y_train


    def add_layers(self):
        """Build the LSTM model with 3 hidden layers, each paired 
        with a 20% dropout layer, and an output layer."""

        self.model.add(LSTM(
            units=30,
            batch_input_shape=(
                1, self.X_train.shape[1], self.X_train.shape[2]),
            return_sequences=True,
            # return_state=True,
            stateful=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(
            units=30,
            batch_input_shape=(
                1, self.X_train.shape[1], self.X_train.shape[2]),
            return_sequences=True,
            # return_state=True,
            stateful=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(
            units=30,
            batch_input_shape=(
                1, self.X_train.shape[1], self.X_train.shape[2]),
            return_sequences=False,
            # return_state=True,
            stateful=True))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(units=1))

        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def fit(self, epochs=30):
        """Fit the LSTM model.

        :param epochs: number of epochs to run, defaults to 30
        :type epochs: int, optional
        """

        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=1,
            verbose=1,
            shuffle=False)


    def predict(self, X_test):
        """Return predictions of fitted model with ModelInput.X_test
        as input.

        :param X_test: test set
        :type X_test: np.array
        :return: predictions
        :rtype: np.array
        """

        return self.model.predict(X_test, batch_size=1)



if __name__ == '__main__':
    # fetch sales data
    sd = SalesData()
    sd.set_sales_data()
    sd.merge_shop_data_to_sales()
    sd.merge_item_data_to_sales()

    # massage and reshape data for input
    mi = ModelInput(sd.monthly_sales)
    # mi.set_shop_groups()
    # mi.date_to_cols(mi.shop_grp_1)
    mi.date_to_cols(mi.data)
    mi.stack_periods()
    mi.train_test_split()
    mi.scale_min_max()
    mi.reshape()

    # build and train model
    model = LSTMModel(mi.X_train, mi.y_train)
    model.add_layers()
    model.fit(epochs=30)

    # score predictions
    y_pred = model.predict(mi.X_test)
    mi.test_results(y_pred)