# coding=utf-8
from numpy import exp, array, random, dot
import scipy.io as sio
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#input img 28X28 to


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = (2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1)

    def pooling_max(self,inputdata):

        row = inputdata.Rank

        col =inputdata.GetLength(1)
        lock2=[0]*row
        lock=lock2*row
        now=[0]*4
        for x in xrange(row):
            for y in xrange(col):
                if(x%2==0 and y%2==0):
                    now[0] = inputdata[x][y]
                    now[1] = inputdata[x-1][y]
                    now[2] = inputdata[x][y-1]
                    now[3] = inputdata[x-1][y-1]
                    lock[x/2][y/2]=max(now)



        return lock



    def __init__(self, layer1, layer2, layer3):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3


    # Sigmoid函数，S形曲线
    # 传递输入的加权和，正规化为0-1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __ReLU(self,x):
        c=x.copy()
        c=array(c)
        c[c<0]=0.1
        c[c>0]=1
        z=c*x
        return z


    def __ReLU_derivative(self, x):
        x[x <= 0] = 0.1
        x[x > 0] = 1
        return x


    # Sigmoid函数的导数，Sigmoid曲线的梯度，表示对现有权重的置信程度
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    def cutnum(self,training_set_output , output_from_laye,test_num_):
        output=[1]*test_num_
        for x in xrange (test_num_):
            output[x]=training_set_outputs[x]-output_from_laye[x][0]
        return output
    def num_x(self,a,b,test_num_):
        outpu = [1] * test_num_

        for x in xrange(test_num_):
            outpu[x]= a[x]*b[x]
        return array(outpu)

    # 通过试错训练神经网络，每次微调突触权重
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations,test_num,weight_decays):
        for iteration in xrange(number_of_training_iterations):
            # 将整个训练集传递给神经网络
            if iteration%5000==0:
                weight_decay=0.1*weight_decays
            output_from_layer_1, output_from_layer_2, output_from_layer_3= self.think(
                training_set_inputs)
            layer3_error = training_set_outputs-output_from_layer_3
            layer3_delta = layer3_error * self.__ReLU_derivative(output_from_layer_3)
            # 计算第9层的误差
            # layer3_error = self.cutnum(training_set_outputs , output_from_layer_3,test_num)
            # test =self.__ReLU_derivative(output_from_layer_3)
            # layer3_delta = self.num_x(layer3_error ,test,test_num)
            # 计算第二层的误差
            layer2_error = layer3_delta.dot(self.layer3.synaptic_weights.T)
            layer2_delta = layer2_error * self.__ReLU_derivative(output_from_layer_2)
            # 计算第一层的误差，得到第一层对第二层的影响
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__ReLU_derivative(output_from_layer_1)

            # 计算权重调整量
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)*weight_decay
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)*weight_decay
            layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)*weight_decay


            # 调整权重
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment
            self.layer3.synaptic_weights += layer3_adjustment

    def cov1(self, inputdata,nuclear, bias):
        array = [0] * 26
        arrays = [array] * 26
        cov = arrays * 1
        for x in xrange(1):
             for a in xrange(26):
                for b in xrange(26):
                     sr = np.multiply(inputdata[x][a:a + 2][b:b + 2], nuclear)
                    cov[x][a][b] = sum(sr) - bias

        return cov
    # 神经网络一思考

    def think(self, inputdata,nuc1,bias1,nuc2,bias2):
        array = [0] * 26
        arrays = [array] * 26
        z=[0]*13
        z1=z*13
        pool1_output=z1*60
        cov1_output = arrays * 60
        cov2_output = arrays * 60


        for x in xrange (60):
            arrays=cov1(inputdata,nuc1(x),bias1(x))
            pool1_output(x)=max_pooling(arrays)
            for y in xrange(60):
            cov2_output(x) = cov2(inputdata, nuc2(x), bias2(x))
            pool2_output(x*60+y) = max_pooling(arrays)


        output_from_layer1 = self.__ReLU(dot( pool2_output, self.layer1.synaptic_weights))
        output_from_layer2 = self.__ReLU(dot(output_from_layer1, self.layer2.synaptic_weights))
        output_from_layer3 = self.__ReLU(dot(output_from_layer2, self.layer3.synaptic_weights))
        return output_from_layer1, output_from_layer2, output_from_layer3


    # 输出权重
    def print_weights(self):
        print "    Layer 1 (8 neurons, each with 3 inputs): "
        print self.layer1.synaptic_weights
        print "    Layer 2 (12 neuron, with  8 inputs):"
        print self.layer2.synaptic_weights

        print "    Layer 3 (4 neurons, each with 12 inputs): "
        print self.layer3.synaptic_weights



if __name__ == "__main__":
    # 设定随机数种子
    random.seed(1)

    # 创建第1层 (8神经元, 每个6输入)
    layer1 = NeuronLayer(1000,3600)
    # 创建第2层 (6神经元，8输入)
    layer2 = NeuronLayer(50, 1000)
    # 创建第3层 (6神经元，8输入)
    layer3 = NeuronLayer(1, 50)



    # 组合成神经网络
    neural_network = NeuralNetwork(layer1, layer2, layer3)

    print "Stage 1) 随机初始突触权重： "
    neural_network.print_weights()

    # 训练集，7个样本，均有3输入1输出


    """


    data = sio.loadmat('9X9.mat')
    inputs= data['is']
    label=sio.loadmat('label.mat')
    outputs=label['label']
    training_set_inputs = [1] *2000
    training_set_outputs=[1] * 2000
    test_set_inputs=[1]*1000
    test_set_outputs=[1]*1000
    for x in xrange(2000):

        training_set_inputs[x] = inputs[x]
        if outputs[x]==1:
            training_set_outputs[x]=1
        if outputs[x]!=1:
            training_set_outputs[x] = 0
    for y in xrange(5000,6000):
        test_set_inputs[y-5000] = inputs[y]
        test_set_outputs[y-5000] = outputs[y]
    """
    # 用训练集训练神经网络
    # 迭代60000次，每次微调权重值
    neural_network.train(array(training_set_inputs), array(training_set_outputs),10000 ,16,0.0008)

    print "Stage 2) 训练后的新权重值： "
    neural_network.print_weights()
    sum=0
    # 用新数据测试神经网络
    print "Stage 3) 测试网络 [1, 1, 0,1] -> ?: "
    hidden_state1, hidden_state2, output = neural_network.think(  array([1, 0, 1, 0, 1, 1,0]))
    print hidden_state1, hidden_state2, output
