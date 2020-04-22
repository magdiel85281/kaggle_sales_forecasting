import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sales_data import SalesData


class LinReg(object):

    def __init__(self):
        self.lr = LinearRegression(fit_intercept=True, n_jobs=2)
        self.X = None
        self.X_range = None
        self.y = None
        self.intercept = None
        self.coeff = None

    def fit(self, X, y):
        self.X = X
        self.X_range = np.array(np.arange(X.shape[0]).reshape(-1, 1))
        self.y = y
        self.lr.fit(self.X_range, self.y)
        self.intercept = self.lr.intercept_
        self.coeff = self.lr.coef_

    def predict_next(self):
        return self.lr.predict(
            np.array(self.X_range.max() + 1).reshape(-1, 1))[0]


class PolyReg(object):

    def __init__(self, degree):
        self.lr = LinearRegression(fit_intercept=True, n_jobs=2)
        self.pr = PolynomialFeatures(degree=degree)
        self.X = None
        self.X_range = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.X_range = np.array(np.arange(X.shape[0]).reshape(-1, 1))
        self.y = y

        X_p = self.pr.fit_transform(self.X_range)
        self.pr.fit(X_p, self.y)
        self.lr.fit(X_p, self.y)

    def predict_next(self):
        return self.lr.predict(
            self.pr.fit_transform(
                np.array(self.X_range.max() + 1).reshape(-1, 1)))[0]


class MultiReg(object):

    def __init__(self, poly_degree):
        self.mlr = LinReg()
        self.mpr = PolyReg(degree=poly_degree)
        self.X = None
        self.X_range = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.X_range = np.array(np.arange(X.shape[0]).reshape(-1, 1))
        self.y = y

        self.mlr.fit(X, y)
        self.mpr.fit(X, y)

    def predict_next(self):
        lr_y_pred = self.mlr.predict_next()
        pr_y_pred = self.mpr.predict_next()
        return np.mean([lr_y_pred, pr_y_pred])

    def plot(self):
        m = self.mlr.coeff[0]
        b = self.mlr.intercept

        fig = plt.figure(figsize=(16, 8))
        plt.plot(self.X_range, self.y) #, color='red')
        plt.plot(self.X_range, m * self.mlr.X_range + b) #, color='blue')
        plt.plot(self.X_range, self.mpr.lr.predict(
            self.mpr.pr.fit_transform(self.mpr.X_range)))
        plt.show


def shop_item(df, shop, item):
    return df.loc[(df['shop_id'] == shop) & (df['item_id'] == item)].copy()


def format_model_input(df):
    # format plot_date as datetime
    df['plot_date'] = df['month'].astype(str) + '-' + \
        df['year'].astype(str)
    df['plot_date'] = df['plot_date'].apply(
        lambda x: datetime.strptime(x, '%m-%Y'))

    mdl_input = df.loc[:, ['plot_date', 'item_cnt_month']].copy()
    mdl_input = mdl_input.set_index('plot_date')  
    mdl_input = sd.add_missing_periods(mdl_input)
    return mdl_input  


def item_avg(df, item, shop):
    item_month_df = df.loc[
        (df['month'] == '11') & (df['item_id'] == item)]
    if item_month_df.shape[0] > 0:
        print('\t\tUsing Nov all shops avg item count.')
        avg_item_cnt = item_month_df['item_cnt_month'].mean()
    else: 
        item_df = df.loc[(df['item_id'] == item)]
        if item_df.shape[0] > 0:
            print('\t\tUsing all time all shops avg item count.')
            avg_item_cnt = item_df['item_cnt_month'].mean()
        else:
            shop_month_df = df.loc[
                (df['month'] == '11') & (df['shop_id'] == shop)]
            if shop_month_df.shape[0] > 0:
                print('\t\tUsing Nov shop all item avg count.')
                avg_item_cnt = shop_month_df['item_cnt_month'].mean()
            else:
                shop_df = df.loc[(df['shop_id'] == shop)]
                print('\t\tUsing shop all item avg count.')
                avg_item_cnt = shop_df['item_cnt_month'].mean()
    return avg_item_cnt


if __name__ == '__main__':
    # get monthly sales data
    sd = SalesData()
    sd.set()
    sd.monthly_agg()
    sd.calc_sales
    sales = sd.monthly_sales

    # import test set and iterate through it to get predictions
    test_set = pd.read_csv('data/test.csv')
    test_set['shop_id'] = test_set['shop_id'].astype(str)
    test_set['item_id'] = test_set['item_id'].astype(str)
    initial_count = test_set.shape[0]

    output_list = []

    for row in range(test_set.shape[0]):
    # for row in range(20):
        row_id = test_set.iloc[row, 0]
        shop_id = test_set.iloc[row, 1]
        item_id = test_set.iloc[row, 2]

        # print(row_id, shop_id, item_id)

        shop_item_df = shop_item(sales, shop_id, item_id)
        if shop_item_df.shape[0] > 0:
            model_input = format_model_input(shop_item_df)

            X = model_input.index
            y = model_input.iloc[:, 0]
            mr = MultiReg(poly_degree=2)
            mr.fit(X, y)
            y_pred = mr.predict_next()

            if y_pred > 0:
                output_list.append({'ID': row_id, 'item_cnt_month': y_pred})
            else: 
                output_list.append({'ID': row_id, 'item_cnt_month': 0})

        else: 
            print(f'{row}: New shop-item data for shop {shop_id} and item {item_id}')
            avg_item_cnt = item_avg(sales, item_id, shop_id)
            print(f'\t\tAvg count: {avg_item_cnt}')
            output_list.append({'ID': row_id, 'item_cnt_month': avg_item_cnt})

        if row % 1000 == 0:
            print('\n***\n***\n\t{:.1f}% complete...\n***\n***\n'\
                .format(row * 100 / initial_count))

    output = pd.DataFrame.from_records(data=output_list)
    output.to_csv('output/submission_mean_reg.csv', index=False)

