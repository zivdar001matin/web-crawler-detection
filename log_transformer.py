import numpy as np
import pandas as pd
import re
import math

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from ua_parser import user_agent_parser

class LogTransformer(TransformerMixin):

    def __init__(self):
        self.df_ix = pd.read_csv('datasets/TEHRAN-IX_Public_Traffic_Statistics.csv')
        self.df_ix['Time'] = self.df_ix['Time'].apply(lambda x: x+12)
        self.df_ix['norm_packets'] = MinMaxScaler().fit_transform(self.df_ix['Packets'].values.reshape(-1,1))
        self.df_ix['Time'] = self.df_ix['Time'].values.astype(np.float16)

    def fit(self, X, y=None):
        X_ = X.copy()
        X_['response_length'] = X_['response_length'].apply(lambda x: re.search(r'\d*', x)[0])
        X_.loc[X_['response_length'] == '', 'response_length'] = '0'
        X_['response_length'] = X_['response_length'].values.astype(np.int32)
        self.bins = np.geomspace(1, X_['response_length'].max(), endpoint=True, num=4)

        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = (
            X_  .pipe(self.binning_method)
                .pipe(self.binning_requested_file_type)
                .pipe(self.binning_status_code)
                .pipe(self.binning_response_length)
                .pipe(self.extracting_url_depth)
                .pipe(self.extracting_user_agent_parts)
                # .pipe(self.extracting_packet_time)
                .pipe(self.extracting_timestamp)
                .pipe(self.drop_rest)
        )
        # print(X_)
        return X_
    
    def binning_method(self, df):
        print('binning_method')
        df = df.join(pd.get_dummies(df['method'], sparse=True, prefix='method'))

        columns=['method_Get', 'method_Head', 'method_Options', 'method_Post', 'method_Put']
        for col in columns:
            if col not in df.columns:
                df[col] = 0
                df[col] = df[col].astype(pd.SparseDtype(np.uint8))

        for weight, name in zip([3, 1, 6, 8, 8], columns):
            df[name] = df[name].sparse.to_dense()
            df[name] *= weight
            df[name] = df[name].astype(pd.SparseDtype(np.uint8))

        return df
    
    def binning_requested_file_type(self, df):
        print('binning_requested_file_type')
        df[['requested_file_type']] = df[['requested_file_type']].replace(dict.fromkeys(['png','jpg', 'jpeg','ico', 'gif','svg', 'ic_launcher'], 'img'))
        df[['requested_file_type']] = df[['requested_file_type']].replace(dict.fromkeys(['js','css', 'xml','php'], 'code'))
        df[['requested_file_type']] = df[['requested_file_type']].replace(dict.fromkeys(['html', 'json', 'txt'], 'renderable'))
        df[['requested_file_type']] = df[['requested_file_type']].replace(dict.fromkeys(['apk'], 'app'))
        df[['requested_file_type']] = df[['requested_file_type']].replace(dict.fromkeys(['mp4', 'mov', 'wmv', 'avi', 'flw', 'swf'], 'video'))
        df[['requested_file_type']] = df[['requested_file_type']].replace(dict.fromkeys(['woff2','woff','ttf','eot'], 'font'))
        df[['requested_file_type']] = df[['requested_file_type']].replace(dict.fromkeys([np.nan], 'endpoint'))

        df = df.join(pd.get_dummies(df['requested_file_type'],sparse=True, prefix='requested_file_type'))

        columns = ['img', 'code', 'renderable', 'app', 'video', 'font', 'endpoint']
        for col in columns:
            col = 'requested_file_type_' + col
            if col not in df.columns:
                df[col] = 0
                df[col] = df[col].astype(pd.SparseDtype(np.uint8))

        for weight, name in zip([3, 8, 1, 8, 8, 8, 3], columns):
            name = 'requested_file_type_' + name
            df[name] = df[name].sparse.to_dense()
            df[name] *= weight
            df[name] = df[name].astype(pd.SparseDtype(np.uint8))

        return df

    def binning_status_code(self, df):
        print('binning_status_code')
        df['status_code'] = df['status_code'].apply(lambda x: int(x/100))
        df = df.join(pd.get_dummies(df['status_code'], sparse=True))

        columns=[1, 2, 3, 4, 5]
        for col in columns:
            if col not in df.columns:
                df[col] = 0
                df[col] = df[col].astype(pd.SparseDtype(np.uint8))
        df = df.rename({1: 'is_1xx', 2: 'is_2xx', 3: 'is_3xx', 4: 'is_4xx', 5: 'is_5xx'}, axis=1)

        for weight, name in zip([5, 3, 1, 1, 5], ['1', '2', '3', '4', '5']):
            name = 'is_' + name + 'xx'
            df[name] = df[name].sparse.to_dense()
            df[name] *= weight
            df[name] = df[name].astype(pd.SparseDtype(np.uint8))

        return df
    
    def binning_response_length(self, df):
        print('binning_response_length')
        df['response_length'] = df['response_length'].apply(lambda x: re.search(r'\d*', str(x))[0])
        df.loc[df['response_length'] == '', 'response_length'] = '0'
        df['response_length'] = df['response_length'].values.astype(np.int32)
        df['response_length'] = pd.cut(df['response_length'], bins=self.bins, labels=['small', 'medium', 'big'], include_lowest=True)
        df['response_length'] = df['response_length'].cat.add_categories('zero')
        df['response_length'] = df['response_length'].fillna('zero')
        # df['response_length'].hist(bins=4)

        df = df.join(pd.get_dummies(df['response_length'],sparse=True, prefix='response_length'))

        for weight, name in zip([8, 4, 2, 1], ['zero', 'small', 'medium', 'big']):
            name = 'response_length_' + name
            df[name] = df[name].sparse.to_dense()
            df[name] *= weight
            df[name] = df[name].astype(pd.SparseDtype(np.uint8))

        return df
    
    def extracting_url_depth(self, df):
        print('extracting_url_depth')
        df['url_depth'] = df['url'].apply(lambda x: len(re.findall(r'/', x[1:])))

        return df
    
    def extracting_user_agent_parts(self, df):
        print('extracting_user_agent_parts')
        def find_user_agent_family(s):
            parsed_string = user_agent_parser.Parse(str(s))
            return parsed_string['user_agent']['family']

        def find_os_family(s):
            parsed_string = user_agent_parser.Parse(str(s))
            return parsed_string['os']['family']

        def find_device_brand(s):
            parsed_string = user_agent_parser.Parse(str(s))
            return parsed_string['device']['brand']

        def find_user_agent(df):
            df['user_agent_family'] = df['user_agent'].apply(find_user_agent_family)
            df['os_family'] = df['user_agent'].apply(find_os_family)
            df['device_brand'] = df['user_agent'].apply(find_device_brand)
            return df
        

        def find_from_device_brand(s):
            if (s == 'Spider') or (s is np.nan):
                return s
            else:
                return 'Generic'
        
        df = df.pipe(find_user_agent)

        df[['os_family']] = df[['os_family']].replace(dict.fromkeys(['Android', 'iOS', 'BlackBerry OS', 'Symbian OS', 'Symbian^3', 'KaiOS', 'Windows Phone', 'Tizen'], 'Phone'))
        df[['os_family']] = df[['os_family']].replace(dict.fromkeys(['Mac OS X', 'Windows', 'Ubuntu', 'Linux'], 'PC'))
        # df[['os_family']] = df[['os_family']].replace(dict.fromkeys(['Other'], 'Other'))
        df = df.join(pd.get_dummies(df['os_family'],sparse=True, drop_first=True, prefix='os_family'))
        df = df.drop('os_family', axis=1)

        columns=['os_family_PC', 'os_family_Phone']
        for col in columns:
            if col not in df.columns:
                df[col] = 0
                df[col] = df[col].astype(pd.SparseDtype(np.uint8))


        df['device_brand'] = df['device_brand'].apply(lambda x: find_from_device_brand(x))
        df = df.join(pd.get_dummies(df['device_brand'],sparse=True, drop_first=True, prefix='device_brand'))
        df = df.drop('device_brand', axis=1)

        columns=['device_brand_Spider']
        for col in columns:
            if col not in df.columns:
                df[col] = 0
                df[col] = df[col].astype(pd.SparseDtype(np.uint8))

        # did nothing about it
        df = df.drop('user_agent_family', axis=1)

        return df
    
    def extracting_packet_time(self, df):
        print('extracting_packet_time')
        def get_tehran_ix_weight(time):
            hour = int(time.split('T')[1].split(':')[0])
            minute = int(time.split('T')[1].split(':')[1])
            return self.df_ix.loc[self.df_ix['Time'] == hour+(math.floor(minute/15) * 0.25)]['norm_packets'].iloc[0]
        
        df['time_weight'] = df['time'].apply(lambda x: get_tehran_ix_weight(x))

        return df
    
    def extracting_timestamp(self, df):
        print('extracting_timestamp')
        temp = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S')
        df['timestamp'] = temp.values.astype(np.int64)
        df['timestamp'] = df['timestamp'].apply(lambda x: int(x//1000000000))

        df = df.drop('time', axis=1)

        return df

    def drop_rest(self, df):
        print('drop_rest')
        df = df.drop(['ip', 'user_agent', 'response_time', 'method', 'requested_file_type', 'status_code', 'response_length', 'url', 'timestamp'], axis=1)

        return df
