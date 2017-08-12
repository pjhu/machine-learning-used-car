#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:43:46 2017

@author: pjhu
"""

import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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

    def set_weight(self):
        if self.is_all:
            self.set_all_data_weight()
        else:
            self.set_only_benz_weight()

    # data from tensor
    def set_all_data_weight(self):
        self.w_value_real = np.array([[0.00160868,0.00315033,-0.65369054,
                                       -0.84656468,-0.01299319,0.00918965,
                                       -0.25697834,0.21275834,-0.22439116]])

        self.b_value_real = -0.000589360119097

        self.mean = np.array([1.155118,22.816468,5.441322,
                              3.573024,1.825566,57.131479,
                              0.297067,5569.950494,2.350290,
                              53.662975])

        self.std = np.array([0.362174,24.111786,3.496133,
                             2.090037,0.517305,58.263915,
                             0.631199,4959.169070,2.560058,
                             15.381265])

    def set_only_benz_weight(self):
        self.w_value_real = np.array([[0.04753411,-0.22235307,-0.6768158,
                                       -0.88951429,-0.20945463,0.04892881,
                                       -0.33221391,0.3532882,-0.20678036]])

        self.b_value_real = -0.000763518869085

        self.mean = np.array([1.241816,59.535181,5.778350,
                              3.755137,2.351462,63.319511,
                              0.520733,12743.146224,1.742907,
                              55.731683])

        self.std = np.array([0.428277,34.280915,3.669540,
                             2.063527,0.771241,61.900406,
                             0.866092,7680.792671,2.181059,
                             17.521217])

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


    def predict_price_diff(self):
        # nomalization
        nomalization = np.divide(np.subtract(self.predict_data.iloc[:,2:-1], self.mean[:-1]), self.std[:-1])

        predict_pre = (np.matmul(nomalization, self.w_value_real.transpose())).transpose()[0]
        predict_flaw = np.add(np.multiply(np.add(predict_pre, self.b_value_real), self.std[-1]), self.mean[-1])

        predict_price = np.multiply(np.divide(predict_flaw, 100),self.predict_data['new_car_price'].values)

        # diff predict price with real price
        self.diff = np.divide(np.subtract(predict_price,self.predict_data['price'].values), self.predict_data['price'].values)

    def draw_error_percentage_cumsum(self):
        labels = np.array(['1%','5%','10%','20%','30%','40%','50%','1','1+'])
        scope = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1,100])

        # category
        cut = pd.cut(abs(self.diff), scope, right=False, labels=labels)
        percent = cut.value_counts()

        # plot
        ind = np.arange(10)
        x = labels
        y = np.array([percent['1%'], percent['5%'], percent['10%'], percent['20%'], percent['30%'], percent['40%'], percent['50%'], percent['1'], percent['1+']]).cumsum()

        fig, ax = plt.subplots()
        rects = ax.bar(range(len(y)), y, width=0.8, align='center')
        ax.set_xticks(ind)
        ax.set_xticklabels(x)
        if self.is_all:
            ax.set_ylim(0, self.row_count + 10000)
        else:
            ax.set_ylim(0, self.row_count + 1000)

        ax.set_title('Predicting the Price of Used Cars(Total Cars: {})'.format(self.row_count))
        plt.margins(0.01)

        # def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '{0}\n{1:.2%}'.format(int(height), (height)/self.row_count),
                    ha='center', va='bottom')

        # autolabel(rects)
        red_patch = mpatches.Patch(color='red', label='A/B, A is count, B is percentage')
        red_patch_x = mpatches.Patch(color='red', label='x is percentage of error range')
        red_patch_y = mpatches.Patch(color='red', label='y is number of cars')
        plt.legend(handles=[red_patch, red_patch_x, red_patch_y])
        plt.rcParams["figure.figsize"] = [12.0,8.0]
        plt.show()

    # car used date
    def draw_which_year_sell_car(self):
        labels = np.arange(1,11)
        cut = pd.cut(self.predict_data['use_date'], np.arange(11), right=False, labels=labels)
        inflection = cut.value_counts()

        x = labels
        y = np.divide(inflection.sort_index().cumsum(),self.row_count)

        # fig, ax = plt.subplots()
        # rects = ax.bar(range(len(y)), y, width=0.8, align='center')
        # ax.set_xticks(np.arange(10))
        # ax.set_xticklabels(np.arange(1,11))
        # ax.set_ylim(0, 10000)
        # ax.set_title('Total Cars: {}'.format(len(predict)))
        # plt.margins(0.01)

        # def autolabel(rects):
        #     for rect in rects:
        #         height = rect.get_height()
        #         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
        #                 '{0}\n{1:.2%}'.format(int(height), (height)/len(predict)),
        #                 ha='center', va='bottom')
        # autolabel(rects)
        red_patch = mpatches.Patch(color='red', label='A/B, A is years, B is percentage of cars')
        plt.legend(handles=[red_patch])
        plt.rcParams["figure.figsize"] = [12.0,8.0]
        for ab in zip(np.arange(1,11), y):
            plt.annotate('(%d, %.2f)' % ab, xy=ab, textcoords='data')
        plt.title('Total Cars: {}'.format(self.row_count))
        plt.plot(x,y)
        plt.show()

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
