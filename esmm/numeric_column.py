# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/3/10

# https://bbs.aianaconda.com/thread-520-1-1.html
# input_layer函数支持的输入类型有以下4种：
#  numeric_column特征列。
#  bucketized_column特征列。
#  indicator_column特征列。
#  embedding_column特征列。

import tensorflow as tf
def test_one_column():
    price=tf.feature_column.numeric_column('price') #定义一个特征列
    features={'price':[[1.0],[5.0]]} #将样本数据定义为字典的类型
    net=tf.feature_column.input_layer(features=features,feature_columns=[price]) #传入input_layer函数，生成张量
    with tf.Session() as sess:  # 建立会话输出特征
        tt = sess.run(net)
        return tt
# res=test_one_column()
# print(res)  #array([[1.][5.]]) (2,1)

def test_placeholder_column():
    price=tf.feature_column.numeric_column('price')
    features={'price':tf.placeholder(dtype=tf.float64)} #生成一个value为占位符的字典
    net=tf.feature_column.input_layer(features=features,feature_columns=[price])
    with tf.Session() as sess:  # 建立会话输出特征
        tt = sess.run(net,feed_dict={features['price']:[[1.],[5.]]})
        return tt
# res = test_one_column()
# print(res)  # array([[1.][5.]]) (2,1)

#numeric_column
def test_reshaping():
    tf.reset_default_graph()
    price =tf.feature_column.numeric_column('price', shape=[1, 2])#定义特征列，并指定形状


    features = {'price': [[[1., 2.]], [[5.,6.]]]}      #传入一个三维的数组
    features1 = {'price': [[3., 4.], [7.,8.]]}         #传入一个二维的数组
    net =tf.feature_column.input_layer(features, price)     #生成特征列张量
    net1 = tf.feature_column.input_layer(features1,price)   #生成特征列张量
    with tf.Session() as sess:                             #建立会话输出特征
        print(net.eval())
        print(net1.eval())
# test_reshaping()
# 在代码第31行，创建字典features，传入了一个形状为[2,1,2]的三维数组。这个三维数组中的第一维是数据的条数（2条）；第二维与第三维要与price指定的形状[1,2]一致。无需reshape
# 在代码第32行，创建字典features1，传入了一个形状为[2,2]的二维数组。该二维数组中的第一维是数据的条数（2条）；第二维代表每条数据的列数（每条数据有2列）。可reshape变成[1,2]

#bucketized_column
def test_numeric_cols_to_bucketized():
    price =tf.feature_column.numeric_column('price')   #定义连续值特征列

    #将连续值特征列转化成离散值特征列,离散值共分为3段：小于3、3~5之间、大于5


    price_bucketized =tf.feature_column.bucketized_column( price, boundaries=[3.,5])

    features = {                                           #定义字典类型对象
       'price': [[2.], [6.],[4.]],
    }
    #生成输入张量
    net =tf.feature_column.input_layer(features,[ price,price_bucketized])
    with tf.Session() as sess:                         #建立会话输出特征
        sess.run(tf.global_variables_initializer())
        print(net.eval())
test_numeric_cols_to_bucketized()
# array([[2. 1. 0. 0.]
#  [6. 0. 0. 1.]
#  [4. 0. 1. 0.]])


#indicator_column
def test_numeric_cols_to_identity():
    tf.reset_default_graph()
    price = tf.feature_column.numeric_column('price')#定义连续值特征列

    categorical_column = tf.feature_column.categorical_column_with_identity('price', 6)
    one_hot_style = tf.feature_column.indicator_column(categorical_column)

    features = {                                           #将值传入定义字典
          'price': [[2], [4]],
      }
    #生成输入层张量
    net = tf.feature_column.input_layer(features,[price,one_hot_style])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(net.eval())

test_numeric_cols_to_identity()
# array([[2. 0. 0. 1. 0. 0. 0.]
#        [4. 0. 0. 0. 0. 1. 0.]])

exit()
def test_order():
     tf.reset_default_graph()
     numeric_col = tf.feature_column.numeric_column('numeric_col')
     some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
         'asparse_feature', hash_bucket_size=5)  # 稀疏矩阵，单独放进去会出错
     embedding_col = tf.feature_column.embedding_column(some_sparse_column, dimension=3)
     # 转化为one-hot特征列
     one_hot_col = tf.feature_column.indicator_column(some_sparse_column)
     print(one_hot_col.name)  # 输出one_hot_col列的名称
     print(embedding_col.name)  # 输出embedding_col列的名称
     print(numeric_col.name)  # 输出numeric_col列的名称
     features = {  # 定义字典数据
         'numeric_col': [[3], [6]],
         'asparse_feature': [['a'], ['x']],
     }
     # 生成输入层张量
     cols_to_vars = {}
     net = tf.feature_column.input_layer(features, [numeric_col, embedding_col, one_hot_col], cols_to_vars)
     with tf.Session() as sess:  # 通过会话输出数据
         sess.run(tf.global_variables_initializer())
         print(net.eval())

test_order()

#根据特征列生成交叉列

from tensorflow.python.feature_column.feature_column import _LazyBuilder

def test_crossed():                            #定义交叉列测试函数
    a = tf.feature_column.numeric_column('a', dtype=tf.int32, shape=(2,))
    b = tf.feature_column.bucketized_column(a, boundaries=(0, 1))      #离散值转化
    crossed = tf.feature_column.crossed_column([b, 'c'], hash_bucket_size=5)                                           #生成交叉列

    builder = _LazyBuilder({                           #生成模拟输入的数据
    'a':
       tf.constant(((-1.,-1.5), (.5, 1.))),
    'c':
       tf.SparseTensor(
           indices=((0, 0), (1, 0), (1, 1)),
           values=['cA', 'cB', 'cC'],
           dense_shape=(2, 2)),
    })
    id_weight_pair = crossed._get_sparse_tensors(builder)#生成输入层张量
    with tf.Session() as sess2:                            #建立会话session，取值
        id_tensor_eval = id_weight_pair.id_tensor.eval()
        print(id_tensor_eval)  # 输出稀疏矩阵
        dense_decoded = tf.sparse_tensor_to_dense( id_tensor_eval, default_value =-1).eval(session=sess2)
        print(dense_decoded)                         #输出稠密矩阵


test_crossed()

# 代码第126行用tf.feature_column.crossed_column函数将特征列b和c混合在一起，生成交叉列。该函数有以下两个必填参数。
#  key：要进行交叉计算的列。以列表形式传入（代码中是[b,ꞌcꞌ]）。
#  hash_bucket_size：要散列的数值范围（代码中是5）。表示将特征列交叉合并后，经过hash算法计算并散列成0~4之间的整数。