# coding: utf-8
"""

    TimeUtils.py

    Copyright (c) 2019-2022, Masatsuyo Takahashi, KEK-PF

"""
from datetime import datetime, timedelta
from time import sleep

"""
    from https://stackoverflow.com/questions/538666/format-timedelta-to-string
"""
def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%0d:%02d' % (hours, minutes, seconds)

def seconds_to_datetime(seconds):
    return datetime.strptime(format_seconds(seconds), "%H:%M:%S")

def friendly_time_str(given_time):
    format  = "%Y-%m-%d %H:%M"
    currn_str = datetime.now().strftime(format)
    given_str = given_time.strftime(format)

    hhmm = given_str[11:]
    if given_str[0:10] == currn_str[0:10]:
        ret_str = hhmm
    else:
        if given_str[0:6] == currn_str[0:6]:
            dd1 = int(given_str[8:10])
            dd0 = int(currn_str[8:10])
            n = dd1 - dd0
            if n == 1:
                ret_str = "tomorrow " + hhmm
            else:
                mm1 = int(given_str[5:7])
                ret_str = "%d/%d %s" % (mm1, dd1, hhmm)
        else:
            ret_str = given_str
    return ret_str

def wait_until_hhmm(yyyy_mm_dd_hh_mm, interval=60, debug=False):
    while True:
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        if debug:
            print(datetime_str)
        if datetime_str >= yyyy_mm_dd_hh_mm:
            break
        sleep(interval)

if __name__ == '__main__':
    datetime_str = (datetime.now() + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M")
    wait_until_hhmm(datetime_str, debug=True)
