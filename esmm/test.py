# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/3/10
import tensorflow as tf
import numpy as np
import pandas as pd
writer = tf.python_io.TFRecordWriter('data/%s.tfrecord' %'data')
#behaviorBids,behaviorC1ids,behaviorCids,behaviorSids,behaviorPids,
#productId,sellerId,brandId,cate1Id,cateId,
#matchScore,popScore,brandPrefer,cate2Prefer,catePrefer,sellerPrefer,matchType,position
#triggerNum,triggerRank,type,hour,phoneBrand,phoneResolution,phoneOs,tab
data_dict={'behaviorBids':[0,1,2,3,4,5,6],'behaviorC1ids':[1,2,3,4,5,6,7],'behaviorCids':[1,2,3,4,5,6,8],
           'behaviorSids':[1,2,3,4,5,6,7],'behaviorPids':[1,2,3,4,5,6,7],
           'productId':[7,6,5,4,3,2,1],'sellerId':[1,2,3,4,5,6,7],'brandId':[1,2,3,4,5,6,7],
           'cate1Id':[1,2,3,4,5,6,7],'cateId':[1,2,3,4,5,6,7],
           'matchScore':[1,2,3,4,5,6,7],'popScore':[1,2,3,4,5,6,7],'brandPrefer':[1,2,3,4,5,6,7],
           'cate2Prefer':[1,2,3,4,5,6,7],'catePrefer':[1,2,3,4,5,6,7],'sellerPrefer':[1,2,3,4,5,6,7],
           'matchType':[1,1,1,2,2,2,3],'position':[2,2,2,3,3,3,1],'triggerNum':[1,1,1,2,2,2,3],
           'triggerRank':[2,2,2,3,3,3,1],'type':[4,4,3,3,1,1,1],'hour':[8,9,10,11,12,1,2],
           'phoneBrand':[b'apple',b'sum',b'huawei',b'xiaomi',b'apple',b'xiaomi',b'huawei'],
           'phoneResolution':[b'a',b'b',b'c',b'd',b'e',b'f',b'g'],
           'phoneOs':[b"android", b"ios",b"android", b"ios",b"android", b"ios",b'ko'],
           'tab':[b"ALL", b"TongZhuang", b"XieBao", b"MuYing", b"NvZhuang", b"MeiZhuang", b"JuJia"]
           }
lg=len(data_dict.keys())
data_df=pd.DataFrame(data_dict)
columns_nm_list=list(data_df.columns)

for i in range(len(data_df)):
    # 创建字典
    features = {}
    # 写入标量，类型Int64，由于是标量，所以"value=[scalars[i]]" 变成list
    features[columns_nm_list[0]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[0]][i]]))
    features[columns_nm_list[1]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[1]][i]]))
    features[columns_nm_list[2]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[2]][i]]))
    features[columns_nm_list[3]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[3]][i]]))
    features[columns_nm_list[4]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[4]][i]]))
    features[columns_nm_list[5]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[5]][i]]))
    features[columns_nm_list[6]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[6]][i]]))
    features[columns_nm_list[7]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[7]][i]]))
    features[columns_nm_list[8]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[8]][i]]))
    features[columns_nm_list[9]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[9]][i]]))
    features[columns_nm_list[10]] = tf.train.Feature(float_list=tf.train.FloatList(value=[data_df[columns_nm_list[10]][i]]))
    features[columns_nm_list[11]] = tf.train.Feature(float_list=tf.train.FloatList(value=[data_df[columns_nm_list[11]][i]]))
    features[columns_nm_list[12]] = tf.train.Feature(float_list=tf.train.FloatList(value=[data_df[columns_nm_list[12]][i]]))
    features[columns_nm_list[13]] = tf.train.Feature(float_list=tf.train.FloatList(value=[data_df[columns_nm_list[13]][i]]))
    features[columns_nm_list[14]] = tf.train.Feature(float_list=tf.train.FloatList(value=[data_df[columns_nm_list[14]][i]]))
    features[columns_nm_list[15]] = tf.train.Feature(float_list=tf.train.FloatList(value=[data_df[columns_nm_list[15]][i]]))
    features[columns_nm_list[16]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[16]][i]]))
    features[columns_nm_list[17]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[17]][i]]))
    features[columns_nm_list[18]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[18]][i]]))
    features[columns_nm_list[19]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[19]][i]]))
    features[columns_nm_list[20]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[20]][i]]))
    features[columns_nm_list[21]] = tf.train.Feature(int64_list=tf.train.Int64List(value=[data_df[columns_nm_list[21]][i]]))
    features[columns_nm_list[22]] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_df[columns_nm_list[22]][i]]))
    features[columns_nm_list[23]] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_df[columns_nm_list[23]][i]]))
    features[columns_nm_list[24]] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_df[columns_nm_list[24]][i]]))
    features[columns_nm_list[25]] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_df[columns_nm_list[25]][i]]))

    # # 写入向量，类型float，本身就是list，所以"value=vectors[i]"没有中括号
    # features['vector'] = tf.train.Feature(float_list=tf.train.FloatList(value=vectors[i]))
    #
    # # 写入矩阵，类型float，本身是矩阵，一种方法是将矩阵flatten成list
    # features['matrix'] = tf.train.Feature(float_list=tf.train.FloatList(value=matrices[i].reshape(-1)))
    # # 然而矩阵的形状信息(2,3)会丢失，需要存储形状信息，随后可转回原形状
    # features['matrix_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=matrices[i].shape))

    # # 写入张量，类型float，本身是三维张量，另一种方法是转变成字符类型存储，随后再转回原类型
    # features['tensor'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensors[i].tostring()]))
    # # 存储丢失的形状信息(806,806,3)
    # features['tensor_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=tensors[i].shape))
# 将存有所有feature的字典送入tf.train.Features中
tf_features = tf.train.Features(feature= features)
# 再将其变成一个样本example
tf_example = tf.train.Example(features = tf_features)
# 序列化该样本
tf_serialized = tf_example.SerializeToString()

# 写入一个序列化的样本
writer.write(tf_serialized)
# 由于上面有循环3次，所以到此我们已经写了3个样本
# 关闭文件
writer.close()