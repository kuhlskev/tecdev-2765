#! /usr/bin/env python3

'''
mamikhai@cisco.com
11/23/2019
monitor-n18g.py
add awareness of weekdays/weekends
10/09/2019
Now this is working great. with normalized features to start from 0 each. with no overlap between features last hour and label/target hour. And with training target (label) being same hour previous day.
Next: will look at:
    1. changing curves to be rate instead of incremental.
    2. checking the best training approach/sequence.
10/03/2019
Will test changing training on 20 minutes previous to a day earlier!

notes from previous release -n18d:
10/01/2019
Previous monitor-n18c.py model is working fine. Here will move to a couple of changes next:
    1. see if model layers, shape can make better prediction of shape, and differentiate shape for each output dimension.
    2. for that be able to reduce the cycle overlap from 50 minutes to like 30m.
10/03/2019
Will put this to work on the server... It is showing tighter fitting to training but a bit wider margin to validation than monitor-n18a.py. Like 114k/138k f
or this versus 129/131 for -n18a. Maybe because of having 20 minute gap versus only 10. Or because of many more hidden nodes, or both. But I like the idea o
f longer gap between training and validation.
'''

import math
import pandas as pd
from sklearn import metrics
from influxdb import DataFrameClient
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
import time
from datetime import datetime
from matplotlib import pyplot as plt
import os.path

pd.options.display.max_rows = None
pd.options.display.max_columns = None

model_directory = './modeln18g' # delete or rename folder to have a fresh model, or if you change layers, optimizer, etc.

x_periods = 0

feature_mean = 0.0
feature_std = 0.0
feature_max = 0.0

client = DataFrameClient(host='localhost', port=8086, database='mdt_db')

tunnel_ifs = ['tunnel-te11200', 'tunnel-te11201', 'tunnel-te13501', 'tunnel-te13502', 'tunnel-te13703', 'tunnel-te13704', 'tunnel-te17801', 'tunnel-te17802', 'tunnel-te12400', 'tunnel-te12401', 'tunnel-te12402', 'tunnel-te12403', 'tunnel-te12404', 'tunnel-te12500', 'tunnel-te12501', 'tunnel-te12502', 'tunnel-te12503', 'tunnel-te12504']

# physical_ifs = ['GigabitEthernet0/0/0/0.1224', 'GigabitEthernet0/0/0/0.1424', 'GigabitEthernet0/0/0/0.1225', 'GigabitEthernet0/0/0/0.1525']
physical_ifs = []

def read_data(field_key, measurement_name, condition1, condition2, condition3, limit, label):
  query_db = str('SELECT "%s" FROM "%s" WHERE %s AND %s AND %s LIMIT %d ' % (field_key, measurement_name, condition1, condition2, condition3, limit+1))
  data_db = client.query(query_db)
  data_df = pd.DataFrame(data_db[str(measurement_name)])
  data_df.columns = [label]
  data_df.reset_index(drop=True, inplace=True)
  data_df.fillna(method='ffill', inplace=True)
  data_df.fillna(method='bfill', inplace=True)
  data_df -= data_df.min()
  data_df.drop(data_df.index[0], inplace=True)
  # data_df = data_df.sub(data_df.shift(fill_value=0))
  # print('\n', query_db, '\n', data_df.describe())
  return data_df

def read_last_target(record_count, label_prefix, verbose=True):
    # read last 1h of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        label = str(label_prefix + interface[-7:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1h - 1m', 'time <= now()', record_count, label) 
        if interface == tunnel_ifs[0]:
            validate_target = read_if
        else:
            validate_target = pd.concat([validate_target, read_if], axis=1, sort=False)
    validate_target.fillna(method='ffill', inplace=True)
    '''
    # read last 1h of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1h - 1m', 'time <= now()', record_count, label)
        validate_target = pd.concat([validate_target, read_if], axis=1, sort=False)
    
        label = str(label_prefix + 'r' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1h - 1m', 'time <= now()', record_count, label)
        validate_target = pd.concat([validate_target, read_if], axis=1, sort=False)
    '''
    if verbose:
        print('\nvalidation target')
        print(validate_target.describe())
    return validate_target

def read_validate(record_count, label_prefix, verbose=True):
    # read previous 1h, same hour previous day, same hour a week ago, of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        query_if = str('("interface-name" = \'%s\')' % (interface))
        label = str(label_prefix + interface[-7:] + "_previous")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2h - 1m', 'time <= now()', record_count, label)    # - 90m for 30 minute overlap
        if interface == tunnel_ifs[0]:
            validate = read_if
        else:
            validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1d")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 1h - 1m'.format(previous), 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_2w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2w - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
    validate.fillna(method='ffill', inplace=True)
    
    '''
    # read previous 1h, same hour previous day, same hour a week ago, of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
    '''
    if verbose:
        print('\nvalidation data')
        print(validate.describe())
    return validate

def read_train_target(record_count, label_prefix, verbose=True):
    # read previous day's same 1h of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        label = str(label_prefix + interface[-7:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 1h - 1m'.format(previous), 'time <= now()', record_count, label)    # - 90m for 30 minute overlap
        if interface == tunnel_ifs[0]:
            train_target = read_if
        else:
            train_target = pd.concat([train_target, read_if], axis=1, sort=False)
    train_target.fillna(method='ffill', inplace=True)

    '''
    # read previous day's same 1h of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 1h - 1m', 'time <= now()', record_count, label)
        train_target = pd.concat([train_target, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 1h - 1m', 'time <= now()', record_count, label)
        train_target = pd.concat([train_target, read_if], axis=1, sort=False)
    '''
    if verbose:
        print('\ntraining target')
        print(train_target.describe())
    return train_target

def read_train_target_long(record_count, label_prefix, verbose=True):
    # read previous day's 24h of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        label = str(label_prefix + interface[-7:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 12h - 30m - 1m'.format(previous), 'time <= now()', record_count, label)
        if interface == tunnel_ifs[0]:
            train_target = read_if
        else:
            train_target = pd.concat([train_target, read_if], axis=1, sort=False)
    
    train_target.fillna(method='ffill', inplace=True)
    '''
    # read previous day's 24h of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train_target = pd.concat([train_target, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train_target = pd.concat([train_target, read_if], axis=1, sort=False)
    '''
    if verbose:
        print('\ntraining target')
        print(train_target.describe())
    return train_target

def read_train(record_count, label_prefix, verbose=True):
    # read previous 1h to target (last day's same 1h), day earlier, a week earlier, of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        query_if = str('("interface-name" = \'%s\')' % (interface))
        label = str(label_prefix + interface[-7:] + "_previous")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 2h - 1m'.format(previous), 'time <= now()', record_count, label)
        if interface == tunnel_ifs[0]:
            train = read_if
        else:
            train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1d")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 1d - 1h - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - {} - 1h - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_2w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2w - {} - 1h - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
    
    train.fillna(method='ffill', inplace=True)
    '''
    # read previous 1h to target (last day's same 1h), day earlier, a week earlier, of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 2h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 2h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2d - 1h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2d - 1h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1d - 1h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1d - 1h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
    '''
    return train

def read_train_long(record_count, label_prefix, verbose=True):
    global feature_mean
    global feature_std
    global feature_max
    # read 24h shifted 1h to target,  previous 24h to target, same day a week earlier, of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        query_if = str('("interface-name" = \'%s\')' % (interface))
        label = str(label_prefix + interface[-7:] + "_previous")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 13h - 30m - 1m'.format(previous), 'time <= now()', record_count, label)   # was - 20h - 30m
        if interface == tunnel_ifs[0]:
            train = read_if
        else:
            train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1d")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 1d - 12h - 30m - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - {} - 12h - 30m - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_2w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2w - {} - 12h - 30m - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
    train.fillna(method='ffill', inplace=True)

    '''
    # read 24h shifted 1h to target, previous 24h to target, same day a week earlier, of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 13h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 13h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
    '''
    if feature_mean == 0:
        feature_mean = train.mean().mean()
        print('feature mean: ', feature_mean)
    if feature_std == 0:
        feature_std = train.std().mean()
        print('feature std: ', feature_std)
    if feature_max == 0:
        feature_max = train.max().mean() / 24 # The mean max per 1 hour
        print('feature max: ', feature_max)
    if verbose:
        print('\ntraining long data')
        print(train.describe())
    return train


def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 

  # epsilon = 0.000001
  epsilon = 0.0

  # choose best normalization of input data
  '''
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val - feature_mean) / (feature_std))
              for my_feature in input_features])
  
  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val - input_features[my_feature].mean()) / (input_features[my_feature].std()))
              for my_feature in input_features])
  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val) / (input_features[my_feature].max()))
              for my_feature in input_features])
  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val - input_features[my_feature].mean()) / (input_features[my_feature].max()))
              for my_feature in input_features])
  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val - int(input_features.mean()[my_feature])) / (int(input_features.std()[my_feature])))
              for my_feature in input_features])
  '''
  # return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val - feature_mean) / (feature_std))
  #             for my_feature in input_features])
  return set([tf.feature_column.numeric_column(my_feature, normalizer_fn=lambda val: (val) / (feature_max))
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=False, num_epochs=None):
    """Trains a neural net regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                             
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_nn_regression_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    if_plot = True,
    prediction = False,
    verbose = True):
  """Trains a neural network regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
    training_examples: A `DataFrame` containing one or more columns 
      to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column
      to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns
      to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column
      to use as target for validation.
      
  Returns:
    A `DNNRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a DNNRegressor object.
  my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer,
      model_dir= model_directory,
      label_dimension= len(tunnel_ifs) + len(physical_ifs)
  )
  
  # Create input functions.
  # print(training_targets)
  # print(training_examples)
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets, 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets, 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets, 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    # training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    # training_predictions = np.array([[item['predictions'][0], item['predictions'][1], item['predictions'][2], item['predictions'][3], item['predictions'][4], item['predictions'][5], item['predictions'][6], item['predictions'][7], item['predictions'][8], item['predictions'][9]] for item in training_predictions])
    training_predictions = np.array([[item['predictions'][i] for i in range(0, len(tunnel_ifs) + len(physical_ifs))] for item in training_predictions])

    if verbose:
        print('training_predictions baoundaries')
        print(training_predictions[0])
        print(training_predictions[len(training_predictions)-1])

    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    # validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    # validation_predictions = np.array([[item['predictions'][0], item['predictions'][1], item['predictions'][2], item['predictions'][3], item['predictions'][4], item['predictions'][5], item['predictions'][6], item['predictions'][7], item['predictions'][8], item['predictions'][9]] for item in validation_predictions])
    validation_predictions = np.array([[item['predictions'][i]for i in range(0, len(tunnel_ifs) + len(physical_ifs))] for item in validation_predictions])

    if verbose:
        print('validation_predictions boundaries')
        print(validation_predictions[0])
        print(validation_predictions[len(validation_predictions)-1])
    
    '''
    # validation plot: if you want to plot every period (slow)
    if if_plot:
        fig = plt.figure(2, figsize=[14, 7])
        fig.clear()
        fig.suptitle('bytes-sent count vs. records', fontsize=16)
        fig_rows = 3
        fig_cols = int((len(tunnel_ifs) + len(physical_ifs)) / fig_rows)
        ifs_plots = fig.subplots(fig_rows, fig_cols, sharex='all', gridspec_kw={'hspace':0.1, 'wspace':0.3, 'left':0.04, 'right':0.99, 'top':0.93, 'bottom':0.03})
        for dim in range(0, len(tunnel_ifs) + len(physical_ifs)):
            ifs_plot = ifs_plots[int(dim / fig_cols), dim % fig_cols]
            ifs_plot.plot(validation_targets.values[:,dim], 'g')
            ifs_plot.plot(validation_predictions[:,dim], 'b')
            ifs_plot.grid(True, which='both', axis='y')
        plt.xticks([0, 59, 119])
        plt.figlegend(["actual", "prediction"], loc='upper right')
        plt.pause(0.0001)
    '''

    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  if if_plot:
    # RMSE values and graphs
    global x_periods
    x_periods += periods
    plt.ion()

    validation_nrmse_prediction = validation_root_mean_squared_error / validation_predictions.mean()
    validation_nrmse_actual = validation_root_mean_squared_error / validation_targets.mean().mean()
  
    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)
    print("Final NRMSE (/prediction, /actual): %0.2f, %0.2f" % (validation_nrmse_prediction, validation_nrmse_actual))
    # print(validation_predictions.mean())
    # print(validation_targets.max().mean())
    # validation plot for each dimension
    fig = plt.figure(1, figsize=[14, 7])
    fig.clear()
    fig.suptitle('bytes-sent count vs. records', fontsize=16)
    fig_rows = 3
    fig_cols = int((len(tunnel_ifs) + len(physical_ifs)) / fig_rows)
    ifs_plots = fig.subplots(fig_rows, fig_cols, sharex='all', gridspec_kw={'hspace':0.1, 'wspace':0.3, 'left':0.04, 'right':0.99, 'top':0.93, 'bottom':0.03})
    # plt.ylabel("bytes-sent")
    # plt.xlabel("records")
    for dim in range(0, len(tunnel_ifs) + len(physical_ifs)):
        ifs_plot = ifs_plots[int(dim / fig_cols), dim % fig_cols]
        ifs_plot.plot(validation_targets.values[:,dim], 'g')
        ifs_plot.plot(validation_predictions[:,dim], 'b')
        ifs_plot.grid(True, which='both', axis='y')
        ifs_plot.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
    plt.xticks([0, 59, 119])
    plt.figlegend(["actual", "prediction"], loc='upper right')
    plt.pause(0.0001)

    # graph of loss metrics over periods
    plt.figure(2, figsize=[10, 4])
    plt.ylabel("RMSE (log scale)")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.yscale('log')
    plt.tight_layout()
    plt.plot(range(x_periods - 10, x_periods), training_rmse, 'm')
    plt.plot(range(x_periods - 10, x_periods), validation_rmse, 'r')
    plt.plot(x_periods - 1, validation_rmse[len(validation_rmse) - 1], 'b*')
    plt.legend(["training", "validation", "prediction"])
    if not prediction:
        plt.plot(x_periods - 1, validation_rmse[len(validation_rmse) - 1], 'r*')
    plt.grid(True, which='both', axis='both')
    plt.pause(0.0001)
 
    # graph of normalized loss metrics over periods
    plt.figure(3, figsize=[10, 4])
    plt.ylabel("NRMSE")
    plt.xlabel("Periods")
    plt.title("Prediction RMSE/mean vs. Periods")
    # plt.yscale()
    plt.tight_layout()
    plt.plot(range(x_periods - 10, x_periods), validation_rmse / validation_predictions.mean(), 'b')
    plt.plot(range(x_periods - 10, x_periods), validation_rmse / validation_targets.mean().mean(), 'g')
    plt.plot(x_periods - 1, validation_rmse[len(validation_rmse) - 1] / validation_predictions.mean(), 'b*')
    plt.plot(x_periods - 1, validation_rmse[len(validation_rmse) - 1] / validation_targets.mean().mean(), 'g*')
    plt.legend(["/prediction", "/actual"])
    plt.grid(True, which='both', axis='both')
    plt.pause(0.0001)

  return dnn_regressor

interval = 600 # wait time between prediction cycles, seconds
# hidden_units = [10, 10]
hidden_units = [72, 36, 18]     # 36, 36 is an overkill!

# set normalization functions to validation (1 hour) ranges
# construct_feature_columns(read_validate(120, 'd_'))

previous = '1d'

if not os.path.exists(model_directory):
    # first training
    dnn_regressor = train_nn_regression_model(
        # learning_rate = 0.1,
        learning_rate = 0.1,
        # steps = 3000,
        steps = 3000,
        # batch_size = 120,
        batch_size = 120,
        hidden_units = hidden_units,
        # hidden_units = [80, 80],
        training_examples = read_train_long(2880, 'd_'),
        training_targets = read_train_target_long(2880, 'l_'),
        validation_examples = read_validate(120, 'd_'),
        validation_targets = read_last_target(120, 'v_')
        )
    
    # time.sleep(300)
    time.sleep(interval / 2)

    # second training
    dnn_regressor = train_nn_regression_model(
        # learning_rate = 0.03,
        learning_rate = 0.03,
        # steps = 3000,
        steps = 3000,
        batch_size = 120,
        hidden_units = hidden_units,
        # hidden_units = [80, 80],
        training_examples = read_train_long(2880, 'd_'),
        training_targets = read_train_target_long(2880, 'l_'),
        validation_examples = read_validate(120, 'd_'),
        validation_targets = read_last_target(120, 'v_')
        )

    # time.sleep(300)
    time.sleep(interval /2)

    # third training
    dnn_regressor = train_nn_regression_model(
        # learning_rate = 0.001,
        learning_rate = 0.003,
        steps = 3000,
        batch_size = 120,
        hidden_units = hidden_units,
        training_examples = read_train_long(2880, 'd_'),
        training_targets = read_train_target_long(2880, 'l_'),
        validation_examples = read_validate(120, 'd_'),
        validation_targets = read_last_target(120, 'v_')
        )

    # time.sleep(300)
    time.sleep(interval / 2)

cycle = 0

# learn = 0.1
while True:
    weekday =  datetime.today().weekday()
    if weekday == 0:
        previous = '3d'
    elif weekday >= 5:
        previous = '7d'
    else:
        previous = '1d'

    # if there're pecularities in the 60 minutes data slices, 
    # would be beneficial to retrain on larger dataset, 
    # not necessarily every cycle
    if cycle % 3 == 0: # run every nth cycle
        dnn_regressor = train_nn_regression_model(
            # learning_rate = 0.0003,
            learning_rate = 0.0001,
            steps = 1000,
            batch_size = 120,
            hidden_units = hidden_units,
            training_examples = read_train_long(2880, 'd_', verbose = False),
            training_targets = read_train_target_long(2880, 'l_', verbose = False),
            validation_examples = read_validate(120, 'd_', verbose = False),
            validation_targets = read_last_target(120, 'v_', verbose = False),
            if_plot = False,
            verbose = False
            )

    # train on previous hour set and history, predict on latest hour
    cycle += 1
    print('cycle number ', cycle)
    dnn_regressor = train_nn_regression_model(
        # learning_rate = 0.0003,
        learning_rate = 0.0003,
        # learning_rate = learn,
        steps = 1000,
        batch_size = 120,
        hidden_units = hidden_units,
        training_examples = read_train(120, 'd_'),
        training_targets = read_train_target(120, 'l_'),
        validation_examples = read_validate(120, 'd_'),
        validation_targets = read_last_target(120, 'v_'),
        prediction = True
        )
    # learn /= 5.0
    # time.sleep(600)
    time.sleep(interval)
