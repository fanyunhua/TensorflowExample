#coding = utf-8
import tensorflow as tf
import mnist_forward
import mnist_backward
from PIL import Image
import numpy as np
#对输入的图片输出预测结果
def pre_pic(picName):
    #打开输入的图片
    img = Image.open(picName)
    #对输入图片大小进行处理   为28*28  Image.ANTIALIAS为消除锯齿
    reIm = img.resize((28,28),Image.ANTIALIAS)
    #将图片变成灰度图 并转换为矩阵
    im_arr = np.array(reIm.convert('L'))
    #除去噪声
    threshold = 52
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255-im_arr[i][j]
            if im_arr[i][j]<threshold:
                im_arr[i][j] = 0
            else: im_arr[i][j] = 255
    #reshape(1,784) reshape为1行784列的矩阵
    nm_arr = im_arr.reshape(1,784)
    #print(nm_arr)
    #将矩阵转为浮点型
    nm_arr = nm_arr.astype(np.float32)
    #将0-255之间的浮点数转位0-1之间的浮点数
    im_ready = np.multiply(nm_arr,1.0/255.0)
    return im_ready
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x,None)
        preValue = tf.argmax(y,1)
        varibale_averags = tf.train.ExponentialMovingAverage(mnist_backward.MOVING)
        varibale_to_restore = varibale_averags.variables_to_restore()
        saver = tf.train.Saver(varibale_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                preValue = sess.run(preValue,feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint filed found")
                return -1
def application():
    test_num  = input("input the number of test pictures:")
    for i in range(int(test_num)):
        testPic = input("the path of test picture")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("the number is: ",preValue)
def main():
    application()
if __name__ == '__main__':
    main()