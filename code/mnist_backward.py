#coding = utf-8
from  tensorflow.examples.tutorials.mnist import  input_data
import tensorflow as tf
import os
import mnist_forward

BATCH_SIZE = 200 #一次喂入多少组数据
LEARN_RATE_BASE = 0.1 #学习率
LEARN_RATE_DECAY = 0.99 #学习衰减率
REGULARIZER = 0.0001 #正则化系数
STEPS = 100000
MOVING = 0.99 #滑动平均衰减率
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'


def backWard(mnist):
    x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x,REGULARIZER)

    gloab_steps = tf.Variable(0,trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem+tf.add_n(tf.get_collection('losses'))

    learn_rate = tf.train.exponential_decay(
        LEARN_RATE_BASE,
        gloab_steps,
        mnist.train.num_examples / BATCH_SIZE,
        LEARN_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss,global_step=gloab_steps)
    ema = tf.train.ExponentialMovingAverage(MOVING,gloab_steps)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name='train')

    save = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            save.restore(sess,ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value,step = sess.run([train_op,loss,gloab_steps],feed_dict={x:xs,y_: ys})

            if i%100==0:
                print("traing ",step,"loss",loss_value)
                save.save(sess,os.path.join(
                    MODEL_SAVE_PATH,MODEL_NAME),
                                global_step=gloab_steps)
def main():
    mnist = input_data.read_data_sets('./data/',one_hot=True)
    backWard(mnist)
if __name__ == '__main__':
    tf.device('/gpu:0,gpu:1')

    main()