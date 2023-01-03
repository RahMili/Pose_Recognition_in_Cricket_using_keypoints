import os.path

import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from src.config import basic_config


class Model():
    def __init__(self, train: bool):
        self.train = train
        self.model = self.loader('ann')

    def loader(self, model_name):
        if model_name == "pose_estimator":
            return self.__pose_estimator_loader()
        elif model_name == "ann":
            return self.__ann_loader()

    def __pose_estimator_loader(self):
        size = basic_config['DEFAULT']['image_size']
        model_name = basic_config['DEFAULT']['pose_estimator_model_name']
        w, h = model_wh(size)
        if w == 0 or h == 0:
            model = TfPoseEstimator(get_graph_path(model_name), target_size=(w, h))
        else:
            model = TfPoseEstimator(get_graph_path(model_name), target_size=(w, h))
        return model

    def __ann_loader(self):
        if self.train:
            input = tf.keras.layers.Input(shape=(54,))
            layer1 = tf.keras.layers.Dense(units=128)(input)
            b1 = tf.keras.layers.BatchNormalization()(layer1)
            a1 = tf.keras.layers.Activation('relu')(b1)
            d1 = tf.keras.layers.Dropout(0.3)(a1)
            layer2 = tf.keras.layers.Dense(units=128)(d1)
            b2 = tf.keras.layers.BatchNormalization()(layer2)
            a2 = tf.keras.layers.Activation('relu')(b2)
            d2 = tf.keras.layers.Dropout(0.3)(a2)
            output = tf.keras.layers.Dense(units=5, activation='softmax')(d2)

            model = tf.keras.models.Model(inputs=input, outputs=output)
            return model
        else:
            if (os.path.exists(basic_config['DEFAULT']['model_path'])):
                model = tf.keras.models.load_model(basic_config['DEFAULT']['model_path'])
                return model
            else:
                self.train = True
                model = self.__ann_loader()
                return model


class Best_Weights(tf.keras.callbacks.Callback):

    def __init__(self):
        self.metric_op=-30.0
        self.weights_op=None
        self.epoch_op=-1
    def on_epoch_end(self,epoch,logs={}):
        if logs['val_acc']>=self.metric_op:
            self.metric_op=logs['val_acc']
            self.epoch_op=epoch
            self.weights_op=self.model.get_weights()
    def on_train_end(self,logs={}):
        self.model.set_weights(self.weights_op)
        print('BEST_EPOCH = {}   BEST_SCORE_ON_VALID_SET = {}'.format(self.epoch_op+1,self.metric_op))



