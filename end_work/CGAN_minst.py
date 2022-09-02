# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 9:33
# @Author  : Qisx
# @File    : CGAN_minst.py
# @Software: PyCharm
import os
import numpy as np

from keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Reshape, Input, Flatten, Embedding
from keras.layers import multiply, Dropout,MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input


class CGAN_Minst():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.image_shape = (self.img_rows, self.img_cols, self.channels)
        # self.image_shape = (28, 28, 1)

        self.dim = 100
        self.classes = 10

        # Adma优化器，动态控制学习率
        # lr：学习率 beta_1：一阶矩估计的指数衰减率 epsilon：防止除以0
        self.lr = 0.0002
        self.beta = 0.5
        # optimizer_Adma = Adam(lr=self.lr, beta_1=self.beta, epsilon=1e-08)
        optimizer_Adma = Adam(0.0002, 0.5)

        # ----判别模型----
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']  # 种类判断的交叉熵损失函数
        self.discriminator = self.discriminator_build()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer_Adma,
                                   metrics=['accuracy']
                                   )  # 训练
        # ----生成模型----
        self.generator = self.generator_build()
        noise = Input(shape=(self.dim,))
        label = Input(shape=(1,))
        imag = self.generator([noise, label])
        self.discriminator.trainable = False  # 判别模型的trainable为False,于训练生成模型
        authenticity, noise_label = self.discriminator(imag)  # 图片放入评价网络
        # 生成模型与判别模型结合
        self.combine = Model([noise, label], [authenticity, noise_label])
        self.combine.compile(loss=losses,
                             optimizer=optimizer_Adma
                             )  # 训练

    def generator_build(self):
        model = Sequential()  # 序列模型

        # ----图片生成----
        model.add(Dense(256, input_dim=self.dim))  # 其输出数组的尺寸为 (*, 256)，模型以尺寸(*, 100) 的数组作为输入
        model.add(LeakyReLU(alpha=0.2))  # 激活函数，alpha控制负数部分线性函数的梯度
        model.add(BatchNormalization())  # BN操作

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))  # 激活函数，alpha控制负数部分线性函数的梯度
        model.add(BatchNormalization())  # BN操作

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))  # 激活函数，alpha控制负数部分线性函数的梯度
        model.add(BatchNormalization())  # BN操作

        # 将生成结果映射到784维上
        # 再reshape成28*28*1
        model.add(Dense(np.prod(self.image_shape), activation='tanh'))
        model.add(Reshape(self.image_shape))

        # ----标签生成----
        # 将一个数字转换为固定尺寸的稠密向量
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.classes, self.dim)(label))
        # 生成n维随机数
        noise = Input(shape=(self.dim,))
        # 标签与随机数对应
        noise_label = multiply([noise, label_embedding])
        image = model(noise_label)

        # 带标签的图片
        return Model([noise, label], image)

    def discriminator_build(self):
        # 全连接层
        model = Sequential()  # 序列模型
        model.add(Flatten(input_shape=self.image_shape))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)
        model.add(Dropout(0.2))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)
        model.add(Dropout(0.2))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)
        model.add(Dropout(0.4))

        label = Input(shape=(1,), dtype='int32')
        image = Input(shape=self.image_shape)

        characteristic = model(image)
        # 判别——真伪
        authenticity = Dense(1, activation='sigmoid')(characteristic)
        # 判别——类别
        label = Dense(self.classes, activation='softmax')(characteristic)

        return Model(image, [authenticity, label])

    def train_model(self, epochs, batch_size=128, save_pre_epoch=200):
        """
        :param epochs: 训练迭代次数
        :param batch_size: 每次送入训练长度
        :param save_pre_epoch: 每save_pre_epoch次生成一张图片
        :return:
        """
        # 数据集加载
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 图片归一化
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = np.expand_dims(x_train, axis=3)  # 升维
        y_train = y_train.reshape(-1, 1)

        authenticity_valid = np.ones((batch_size, 1))  # 真
        authenticity_fake = np.zeros((batch_size, 1))  # 伪

        # ----训练----
        for epoch in range(epochs):
            # ------训练鉴别模型------
            # 随机获得图片与其对应的标签
            index = np.random.randint(0, x_train.shape[0], batch_size)
            images, labels = x_train[index], y_train[index]
            # 生成随机输入
            noise = np.random.normal(0, 1, (batch_size, self.dim))  # 图
            noise_labels = np.random.randint(0, 10, (batch_size, 1))  # 标签
            fake_images = self.generator.predict([noise, noise_labels])
            # 利用真图片训练鉴别模型
            real_loss = self.discriminator.train_on_batch(images, [authenticity_valid, labels])  # 真图片训练
            fake_loss = self.discriminator.train_on_batch(fake_images, [authenticity_fake, noise_labels])
            end_loss = 0.5 * np.add(real_loss, fake_loss)

            # -----训练生成模型-----
            loss_generator = self.combine.train_on_batch([noise, noise_labels], [authenticity_valid, noise_labels])

            if epoch % save_pre_epoch == 0:
                print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
                    epoch, end_loss[0], 100 * end_loss[3], 100 * end_loss[4], loss_generator[0]))
                self.save_image(epoch)

    def save_image(self, epoch):
        noise = np.random.normal(0, 1, (2 * 5, 100))
        noise_labels = np.arange(0, 10).reshape(-1, 1)

        fake_images = self.generator.predict([noise, noise_labels])
        fake_images = 0.5 * fake_images + 0.5

        fig, axs = plt.subplots(2, 5)
        cnt = 0
        for i in range(2):
            for j in range(5):
                axs[i, j].imshow(fake_images[cnt, :, :, 0], cmap='gray')
                axs[i, j].set_title("Digit: %d" % noise_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("CGAN_images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    # 判断有没有生成图片的文件夹，如果没有重新创建
    if not os.path.exists("./CGAN_images"):
        os.makedirs("./CGAN_images")
    cgan = CGAN_Minst()
    cgan.train_model(epochs=20000, batch_size=256, save_pre_epoch=500)
