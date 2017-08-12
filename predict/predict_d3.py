#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:43:46 2017

@author: pjhu
"""

import json
import pandas as pd
import numpy as np


class predict:

    def __init__(self):
        self.w_value_real = None
        self.b_value_real = None
        self.mean = None
        self.std = None
        self.row_count = None
        self.is_all = True
        self.predict_data = None
        self.diff = None

    def init_data(self, w, b, mean, std):
        self.w_value_real = w
        self.b_value_real = b
        self.mean = mean
        self.std = std

    def set_is_all(self, is_all=True):
        self.is_all = is_all

    # read new data, waiting for predicting
    def read_data_from_csv(self, file_name):
        predict_data = pd.read_csv(file_name,
                                   names=["city","brand","gender",
                                          "new_car_price","road_haul",
                                          "use_date","displacement",
                                          "follow","transfer","service_price",
                                          "flaw","price"])
        drop_na = predict_data.dropna(how='any')
        if self.is_all:
            self.predict_data = drop_na
        else:
            self.predict_data = drop_na[drop_na['brand'].str.contains('奔驰')]
        self.row_count = len(self.predict_data)


    # caculate error percentage pridict value and real cars value
    # (pridict_price - real_price) / pridict_price
    def predict_price_diff(self):
        # nomalization
        nomalization = np.divide(np.subtract(self.predict_data.iloc[:,2:-1], self.mean[:-1]), self.std[:-1])

        predict_pre = (np.matmul(nomalization, self.w_value_real.transpose())).transpose()[0]
        predict_flaw = np.add(np.multiply(np.add(predict_pre, self.b_value_real), self.std[-1]), self.mean[-1])

        predict_price = np.multiply(np.divide(predict_flaw, 100),self.predict_data['new_car_price'].values)

        # diff predict price with real price
        self.diff = np.divide(np.subtract(predict_price,self.predict_data['price'].values), self.predict_data['price'].values)

    # error percentage for cumsum of cars
    def draw_error_percentage_cumsum(self):
        labels = np.array(['1%','5%','10%','20%','30%','40%','50%','1','1+'])
        scope = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1,100])

        # category
        cut = pd.cut(abs(self.diff), scope, right=False, labels=labels)
        percent = cut.value_counts()
        print('draw_error_percentage_cumsum:', percent)

        x = labels
        y = np.array([percent['1%'], percent['5%'], percent['10%'], percent['20%'], percent['30%'], percent['40%'], percent['50%'], percent['1'], percent['1+']]).cumsum()
        # with open("diff.json", mode='w+') as out_handler:
        #     out_handler.write(json.dumps(dict(zip(x,y.tolist()))))
        pd.DataFrame({'percentage':x, 'cumsum':y}).to_csv('../data/diff.csv', columns=['percentage', 'cumsum'], index=False)

    # draw_which_year_sell_car
    def draw_which_year_sell_car(self):
        labels = np.arange(1,11)
        cut = pd.cut(self.predict_data['use_date'], np.arange(11), right=False, labels=labels)
        inflection = cut.value_counts()
        
        x = labels
        y = np.divide(inflection.sort_index().cumsum(),self.row_count)
        # print('draw_which_year_sell_car:{}{}'.format(x,y))
        pd.DataFrame({'years':x, 'cumsum':y}).to_csv('../data/which_year_sell_car.csv', columns=['years', 'cumsum'], index=False)

if __name__ == "__main__":
    pre = predict()
    is_all = True
    pre.set_is_all(is_all)
    pre.set_weight()

    file_name = '../predict_data/predict.csv'
    pre.read_data_from_csv(file_name)
    pre.predict_price_diff()
    pre.draw_error_percentage_cumsum()
    pre.draw_which_year_sell_car()
