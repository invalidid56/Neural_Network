import neuralnetwork
import os
import matplotlib.pyplot


def process_data(data):
    result = []
    for record in data:
        record = record.split(',')
        result.append([int(record[0]), [int(i)/256 + 0.01 for i in record[1:]]])
        result[-1][0] = [int(result[-1][0] is i)*0.98+0.01 for i in range(10)]
    return result


def max_index(iterable):
    max_d = max(iterable)
    return iterable.index(max_d)

os.chdir(os.getcwd()+'\Data_set\MNIST')
train_file = open('mnist_train.csv')
train_data = train_file.readlines()
train_file.close()
test_file = open('mnist_test.csv')
test_data = test_file.readlines()
test_file.close()
train_data = process_data(train_data)
test_data = process_data(test_data)

print(len(train_data))

nodes = [
    len(train_data[0][1]),
    200,
    10
]
Bradley = neuralnetwork.FeedFowardNeuralNetwork(nodes)
Bradley.dataset = train_data
Bradley.weight = Bradley.train(
    epoch=5,
    learning_rate=0.1,
)
score = sum([max_index(record[0]) is max_index(list(Bradley.query(record[1])[-1])) for record in test_data])/len(test_data)

print(score)
