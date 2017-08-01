#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:50:46 2017

@author: twer
"""
import json
import re
# from memory_profiler import profile
from line_profiler import LineProfiler


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
    with open(source, encoding="utf-8") as file_handler, open(target, "w") as out_handler:
        while True:
            try:
                line = next(file_handler)
                line_json = json.loads(line)
                line_re = re.findall(r'\d+', line_json["detail"]["use_date"])
                if "年" in line_json:
                    line_json["detail"]["use_date"] = ".".join(line_re).strip()
                else:
                    line_json["detail"]["use_date"] = ".".join(["0", line_re[0]])
                out_handler.writelines([json.dumps(line_json)])

            except (IOError, OSError):
                print("Error opening / processing file")
            except StopIteration:
                print("============= stop error")
                break
            except:
                pass


if __name__ == "__main__":
    source = "/Users/twer/Desktop/data/20170619.json"
    target = "/Users/twer/Desktop/data/20170619_target.json"
    # process_file_differently(source, target)

    lp = LineProfiler()
    # lp.add_function(process_file_differently)
    lp_wrapper = lp(process_file)
    # lp_wrapper = lp(process_file_differently)
    lp_wrapper(source, target)
    lp.print_stats()


