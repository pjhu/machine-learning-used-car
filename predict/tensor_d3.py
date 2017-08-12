#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 19:44:30 2017

@author: pjhu
"""

import csv
import json
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow as tf
from predict_d3 import predict
from os.path import dirname, abspath

class tensor:

    def __init__(self):
        self.w_value_real = None
        self.b_value_real = None
        self.mean = None
        self.std = None
        self.row_count = None
        self.is_all = True
        self.source_data = None

    def set_is_all(self, is_all):
        self.is_all = is_all

    # read data from csv
    def read_data_from_csv(self):
        path = os.path.join(dirname(dirname(abspath(__file__))), 'training_set')
        all_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and os.path.join(path, f).endswith('_target.csv')]
        
        df_from_each_file = (pd.read_csv(f, names=["city","brand","gender","new_car_price","road_haul","use_date","displacement","follow","transfer","service_price","flaw","price"]) for f in all_files)
        concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
        return concatenated_df

    # drop duplicates and NA
    def clean_data(self, data):
        concatenated_df = data
        if self.is_all:
            drop_duplicates = concatenated_df.drop_duplicates()
        else:
            only_benz = concatenated_df[concatenated_df['brand'].str.contains('奔驰')]
            drop_duplicates = only_benz.drop_duplicates()

        # drop duplicates
        drop_na = drop_duplicates.dropna(how='any')
        # reset index
        guazi = drop_na.reset_index(drop=True)
        # caculate derating reate
        guazi["hedge_ratio"] = (guazi["price"] / guazi["new_car_price"])*100
        self.source_data = guazi
        self.row_count = len(self.source_data)

    def output_correlation(self):
        # caculate correlation
        # print(self.source_data.corr("pearson"))
        corr = self.source_data.corr("pearson")
        corr.to_csv('../data/correlation.csv')

    # 1d relation(use_date, hedge_ratio)
    def use_date_and_hedge_ratio(self):
        X = self.source_data.use_date
        Y = self.source_data.hedge_ratio
        X = sm.add_constant(X)
        est = sm.OLS(Y,X)
        est = est.fit()
        self.source_data.to_csv('../data/use_date_and_hedge_ratio.csv', columns=['use_date', 'hedge_ratio'], index=False)

    # 2d relation()
    def draw_hedge_date(self):
        est=smf.ols(formula='hedge_ratio ~ use_date + road_haul',data=self.source_data).fit()
        x_surf, y_surf = np.meshgrid(np.linspace(self.source_data.use_date.min(), self.source_data.use_date.max(), 100),np.linspace(self.source_data.road_haul.min(), self.source_data.road_haul.max(), 100))

        onlyX = pd.DataFrame({'use_date': x_surf.ravel(), 'road_haul': y_surf.ravel()})
        fittedY=est.predict(exog=onlyX)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.source_data.use_date ,self.source_data.road_haul ,self.source_data.hedge_ratio ,c='blue', marker='o', alpha=0.5) # 画出原始数据
        ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='r') # 添加回归线，红色

        ax.set_xlabel('use_date')
        ax.set_ylabel('road_haul')
        ax.set_zlabel('hedge_ratio')
        ax.set_title("Total Cars: {0}".format(self.row_count))
        red_patch_k = mpatches.Patch(color='red', label='Slope: {}'.format(est.params[1]))
        red_patch_b = mpatches.Patch(color='red', label='Intercept: {}'.format(est.params[0]))
        plt.legend(handles=[red_patch_k, red_patch_b])
        plt.show()

    # tensor
    def draw_tensor(self):
        tensor_no_z_score = self.source_data.loc[:, ['gender','new_car_price','road_haul','use_date','displacement','follow','transfer','service_price','flaw','hedge_ratio']]

        tensor = (tensor_no_z_score - tensor_no_z_score.mean())/tensor_no_z_score.std()
        
        g = tf.Graph()
        with g.as_default():
            xs = tf.placeholder(tf.float64, shape=(None, 9))
            w = tf.Variable(tf.random_normal(shape=[9], dtype=tf.float64))
            b = tf.Variable(0.0, dtype=tf.float64)
            ys = xs * w + b
            ys_ = tf.placeholder(tf.float64, shape=(None, 1))

            loss = tf.reduce_mean(tf.square(ys_ - ys))
            train = tf.train.AdamOptimizer(1e-4).minimize(loss)

        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            test_xs = tensor.iloc[:,:-1]
            test_ys = tensor.iloc[:,-1:]

            for step in range(10000):
                batch_ = tensor.sample(n=1000)
                batch_xs = batch_.iloc[:,:-1]
                batch_ys = batch_.iloc[:,-1:]
                sess.run(train, feed_dict={xs: batch_xs, ys_: batch_ys})
                if step % 5000 == 0:
                    loss_value = sess.run(loss, feed_dict={xs: test_xs, ys_: test_ys})
                    print('step: %f, loss: %f' % (step, loss_value))

            self.w_value_real, self.b_value_real = sess.run([w, b])
            print(self.w_value_real, self.b_value_real)
        self.mean = tensor_no_z_score.mean()
        self.std = tensor_no_z_score.std()
        print("mean:\n", self.mean)
        print("std:\n", self.std)
        to_json = {
            "w_value_real": dict(zip(tensor.columns[:-1], self.w_value_real)),
            "b_value_real": self.b_value_real,
            "mean": json.loads(self.mean.to_json()),
            "std": json.loads(self.std.to_json())
        }
        with open("../data/tensor.json", mode='w+') as out_handler:
            out_handler.write(json.dumps(to_json))
    
        # tensor.to_csv('../data/tensor.csv', index=False)

if __name__ == "__main__":
    ts = tensor()
    is_all = True
    data = ts.read_data_from_csv()

    ts.set_is_all(is_all)
    ts.clean_data(data)
    ts.output_correlation()
    ts.use_date_and_hedge_ratio()
    ts.draw_tensor()

    print("="*100)
    # predict
    pre = predict()
    pre.set_is_all(is_all)
    pre.init_data(np.array([ts.w_value_real]), ts.b_value_real, np.array(ts.mean), np.array(ts.std))
    
    file_name = '../predict_data/20170621_predict.csv'
    pre.read_data_from_csv(file_name)
    pre.predict_price_diff()
    pre.draw_error_percentage_cumsum()
    pre.draw_which_year_sell_car()
