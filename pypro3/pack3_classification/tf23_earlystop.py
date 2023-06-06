# EarlyStopping을 별도의 모듈로 작성 후 필요한 곳에 적용하기

from keras.callbacks import EarlyStopping
import tensorflow as tf

class MyEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.25:     # loss가 0.25보다 작으면 조기종료. 0.25는 baseline
            print('\n학습이 조기 종료되었습니다.')
            self.model.stop_training = True
            












