#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:50:46 2017

@author: pjhu
"""
import csv
import json
import os
import re
import multiprocessing as mp
import datetime
from os.path import dirname, abspath


def process_file_differently(source, target):
    """
    Process the file line by line using the file's returned iterator
    """
    print(source)
    with open(source, encoding="utf-8") as file_handler, open(target, encoding='utf-8', mode='w+') as out_handler:
        out_writer = csv.writer(out_handler, delimiter=",")
        a = 0
        while True:
            try:
                line = next(file_handler)
                line_json = json.loads(line)
                detail = line_json["detail"]

                if line_json["msrp"]:
                    new_car_price = '.'.join(re.findall(r'\d+', line_json["msrp"]))
                else:
                    continue

                flaw = ""
                if "瑕疵" in detail["imageList"][-1]["category"]:
                    flaw = len(detail["imageList"][-1]["images"])

                line_re = re.findall(r'\d+', detail["use_date"])
                if "年" in detail["use_date"]:
                    use_date = ".".join(line_re).strip()
                else:
                    use_date = ".".join(["0", line_re[0]])

                out_writer.writerow([detail["domain"], detail["title"].split(" ")[0], detail["gender"],
                                     new_car_price, detail["road_haul"],
                                     use_date, detail["air_displacement"],
                                     detail["follow_num"], detail["transfer_num"],
                                     detail["service_charge"]["service_price"], flaw, detail["price"]])

            except (IOError, OSError):
                print("Error opening / processing file")
            except StopIteration:
                print("============= stop error")
                break
            except Exception as e:
                a = a + 1
                # raise e
                pass
        print("#{0} errors".format(a))


if __name__ == "__main__":
    t1 = datetime.datetime.now()
    source_path = os.path.join(dirname(dirname(abspath(__file__))), 'source')
    target_path = os.path.join(dirname(dirname(abspath(__file__))), 'training_set')
    all_files = [(os.path.join(source_path, f), os.path.join(target_path, f.split('.')[0] + '_target.csv')) for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f)) and os.path.join(source_path, f).endswith('.json')]
    
    pool = mp.Pool(processes=4)
    results = [pool.apply_async(process_file_differently, args=(x,y)) for x,y in all_files]
    output = [p.get() for p in results]
    t2 = datetime.datetime.now()
    delta_time = t2 - t1
    print("use time: {}".format(delta_time.seconds))
    

