# from extract_feature import BertVector
# bv = BertVector()
# bv.encode(['今天天气不错'])
# 训练
from similarity import BertSim
import tensorflow as tf

bs = BertSim()
bs.set_mode(tf.estimator.ModeKeys.TRAIN)
bs.train()