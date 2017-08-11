#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:50:46 2017

@author: pjhu
"""
import csv
import json
import re
# from memory_profiler import profile
# from line_profiler import LineProfiler

# @profile
def read_file_directory(path):
    with open(path) as infile:
        for line in infile.readlines():
            print(line.__len__())
    pass


# @profile
def read_large_file(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data


def process_file(source, target):
    with open(source, encoding="utf-8") as file_handler, open(target, "w") as out_handler:
        for line in read_large_file(file_handler):
            try:
                line_json = json.loads(line)
                line_re = re.findall(r'\d+', line_json["detail"]["use_date"])
                if "年" in line_json:
                    line_json["detail"]["use_date"] = ".".join(line_re).strip()
                else:
                    line_json["detail"]["use_date"] = ".".join(["0", line_re[0]])
                    out_handler.writelines([json.dumps(line_json)])
            except (IOError, OSError):
                print("Error opening / processing file")
            except:
                pass

# @profile
def process_file_differently(source, target):
    """
    Process the file line by line using the file's returned iterator
    """
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

                # out_writer.writerow([detail["gender"], new_car_price, detail["road_haul"],
                #                      use_date, detail["air_displacement"],
                #                      detail["follow_num"], detail["transfer_num"],
                #                      detail["service_charge"]["service_price"], flaw, detail["price"]])

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
    source_1 = "/path/to/source/20170619.json"
    target_1 = "/path/to/training_set/20170619_target.csv"
    source_2 = "/path/to/source/20170620.json"
    target_2 = "/path/to/training_set/20170620_target.csv"
    source_3 = "/path/to/source/20170621.json"
    target_3 = "/path/to/training_set/20170621_target.csv"
    #process_file_differently(source_1, target_1)
    #process_file_differently(source_2, target_2)
    process_file_differently(source_3, target_3)

    # process_file_differently(source, target)

    # lp = LineProfiler()
    # lp_wrapper = lp(process_file)
    # lp_wrapper = lp(process_file_differently)
    # lp_wrapper(source_1, target_1)
    # lp_wrapper(source_2, target_2)
    # lp_wrapper(source_3, target_3)
    # lp.print_stats()
