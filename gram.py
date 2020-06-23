# coding=utf-8
#################################################################
# Code written by Edward Choi (mp2893@gatech.edu)
# For bug report, please contact author using the email address
#################################################################

import sys, random, time, argparse
from collections import OrderedDict
import cPickle as pickle
import numpy as np

import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1


def unzip(zipped):
    new_params = OrderedDict()
    for key, value in zipped.iteritems():
        new_params[key] = value.get_value()
    return new_params


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def get_random_weight(dim1, dim2, left=-0.1, right=0.1):
    return np.random.uniform(left, right, (dim1, dim2)).astype(config.floatX)


def load_embedding(options):
    m = np.load(options['embFile'])
    w = (m['w'] + m['w_tilde']) / 2.0
    return w


def init_params(options):
    params = OrderedDict()

    np.random.seed(0)
    inputDimSize = options['inputDimSize']
    numAncestors = options['numAncestors']
    embDimSize = options['embDimSize']
    hiddenDimSize = options['hiddenDimSize']  # hidden layer does not need an extra space
    attentionDimSize = options['attentionDimSize']
    numClass = options['numClass']

    # embedding初始化，这里可能会加载预训练的embedding
    params['W_emb'] = get_random_weight(inputDimSize + numAncestors, embDimSize)
    if len(options['embFile']) > 0:
        params['W_emb'] = load_embedding(options)
        options['embDimSize'] = params['W_emb'].shape[1]
        embDimSize = options['embDimSize']

    # attention参数随机初始化
    params['W_attention'] = get_random_weight(embDimSize * 2, attentionDimSize)
    params['b_attention'] = np.zeros(attentionDimSize).astype(config.floatX)
    params['v_attention'] = np.random.uniform(-0.1, 0.1, attentionDimSize).astype(config.floatX)

    # GRU的参数
    params['W_gru'] = get_random_weight(embDimSize, 3 * hiddenDimSize)
    params['U_gru'] = get_random_weight(hiddenDimSize, 3 * hiddenDimSize)
    params['b_gru'] = np.zeros(3 * hiddenDimSize).astype(config.floatX)

    # 输出的softmax？
    params['W_output'] = get_random_weight(hiddenDimSize, numClass)
    params['b_output'] = np.zeros(numClass).astype(config.floatX)

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for key, value in params.iteritems():
        tparams[key] = theano.shared(value, name=key)
    return tparams


def dropout_layer(state_before, use_noise, trng, prob):
    proj = T.switch(use_noise,
                    (state_before * trng.binomial(state_before.shape, p=prob, n=1, dtype=state_before.dtype)),
                    state_before * 0.5)
    return proj


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def gru_layer(tparams, emb, options):
    hiddenDimSize = options['hiddenDimSize']
    timesteps = emb.shape[0]
    if emb.ndim == 3:
        n_samples = emb.shape[1]
    else:
        n_samples = 1

    def stepFn(wx, h, U_gru):
        uh = T.dot(h, U_gru)
        r = T.nnet.sigmoid(_slice(wx, 0, hiddenDimSize) + _slice(uh, 0, hiddenDimSize))
        z = T.nnet.sigmoid(_slice(wx, 1, hiddenDimSize) + _slice(uh, 1, hiddenDimSize))
        h_tilde = T.tanh(_slice(wx, 2, hiddenDimSize) + r * _slice(uh, 2, hiddenDimSize))
        h_new = z * h + ((1. - z) * h_tilde)
        return h_new

    Wx = T.dot(emb, tparams['W_gru']) + tparams['b_gru']
    results, updates = theano.scan(fn=stepFn, sequences=[Wx],
                                   outputs_info=T.alloc(numpy_floatX(0.0), n_samples, hiddenDimSize),
                                   non_sequences=[tparams['U_gru']], name='gru_layer', n_steps=timesteps)

    return results


def generate_attention(tparams, leaves, ancestors):
    # 根据embedding来计算attention
    attentionInput = T.concatenate([tparams['W_emb'][leaves], tparams['W_emb'][ancestors]], axis=2)
    mlpOutput = T.tanh(T.dot(attentionInput, tparams['W_attention']) + tparams['b_attention'])
    preAttention = T.dot(mlpOutput, tparams['v_attention'])
    attention = T.nnet.softmax(preAttention)
    return attention


def softmax_layer(tparams, emb):
    nom = T.exp(T.dot(emb, tparams['W_output']) + tparams['b_output'])
    denom = nom.sum(axis=2, keepdims=True)
    output = nom / denom
    return output


def build_model(tparams, leavesList, ancestorsList, options):
    dropoutRate = options['dropoutRate']
    trng = RandomStreams(123)
    use_noise = theano.shared(numpy_floatX(0.))

    x = T.tensor3('x', dtype=config.floatX)
    y = T.tensor3('y', dtype=config.floatX)
    mask = T.matrix('mask', dtype=config.floatX)
    lengths = T.vector('lengths', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    embList = []
    for leaves, ancestors in zip(leavesList, ancestorsList):
        # 根据embedding 计算 attention值
        tempAttention = generate_attention(tparams, leaves, ancestors)
        # 计算加上attention后所得到的embedding
        tempEmb = (tparams['W_emb'][ancestors] * tempAttention[:, :, None]).sum(axis=1)
        embList.append(tempEmb)

    emb = T.concatenate(embList, axis=0)

    # 这里是我们需要进行修改的地方，根据输入x来进行筛选输入
    x_emb = T.tanh(T.dot(x, emb))

    # GRU的计算过程
    hidden = gru_layer(tparams, x_emb, options)
    hidden = dropout_layer(hidden, use_noise, trng, dropoutRate)
    # 这里的mask记录了每一个患者visit的长度，非对应visit直接取0
    y_hat = softmax_layer(tparams, hidden) * mask[:, :, None]

    # ===============添加代码 by Rhys=========================
    # 针对y与y_hat计算metric，这里其实是个数学问题了...
    # 已知y的形式，那么y_hat的形式应该也是对应的
    # 那么我们首先：1.要对y_hat的结果进行排序，找出最大的前k个结果；
    # 2.对每个患者进行统计：出现在前20个中的为1的label的数量，总的label为1的数量
    # 3.统计所有患者的结果，然后取比值作为最终准确率

    logEps = 1e-8
    # 计算单个病人计算交叉熵
    # 这里y的维度是 maxVisitLength * patients * label_class
    # 这里的 * 表示对应位置的值直接相乘
    cross_entropy = -(y * T.log(y_hat + logEps) + (1. - y) * T.log(1. - y_hat + logEps))
    # 先对所有的疾病直接累加，这才是文中式子的计算结果
    # 然后对所有的visit求平均，得到每一个患者的cross entropy
    output_loglikelihood = cross_entropy.sum(axis=2).sum(axis=0) / lengths
    # 针对所有病人，计算总loss
    cost_noreg = T.mean(output_loglikelihood)

    # 为loss加上一个正则项
    if options['L2'] > 0.:
        cost = cost_noreg + options['L2'] * ((tparams['W_output'] ** 2).sum() + (tparams['W_attention'] ** 2).sum() + (
                tparams['v_attention'] ** 2).sum())

    return use_noise, x, y, mask, lengths, cost, cost_noreg, y_hat


def load_data(seqFile, labelFile, timeFile=''):
    # 打开文件
    sequences = np.array(pickle.load(open(seqFile, 'rb')))
    labels = np.array(pickle.load(open(labelFile, 'rb')))
    if len(timeFile) > 0:
        times = np.array(pickle.load(open(timeFile, 'rb')))

    np.random.seed(0)
    dataSize = len(labels)
    # 生成0到dataSize-1的随机排序序列
    ind = np.random.permutation(dataSize)
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)

    # 随机划分训练集、验证集和测试集，因为第一个维度代表每一个患者的数据
    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest + nValid]
    train_indices = ind[nTest + nValid:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]
    train_set_t = None
    test_set_t = None
    valid_set_t = None

    if len(timeFile) > 0:
        train_set_t = times[train_indices]
        test_set_t = times[test_indices]
        valid_set_t = times[valid_indices]

    def len_argsort(seq):
        # 按照seq中每一项的长度进行排序
        # 也就是按每一个patient的visit的次数进行升序排序
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    # 将训练、验证、测试数据分别打包
    train_set = (train_set_x, train_set_y, train_set_t)
    valid_set = (valid_set_x, valid_set_y, valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_t)

    return train_set, valid_set, test_set


def adadelta(tparams, grads, x, y, mask, lengths, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in
                      tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y, mask, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in
             zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

    return f_grad_shared, f_update


def padMatrix(seqs, labels, options):
    # length记录了每一个每一个患者visit的次数
    lengths = np.array([len(seq) for seq in seqs]) - 1
    # 表示有n个患者的样本
    n_samples = len(seqs)
    # 最长的visit次数
    maxlen = np.max(lengths)

    # maxlen：最长的sequence，即最多的访问次数
    # n_samples：表示有n条患者的数据，实际实验中均为100
    # 注意这里的数据形式，改为了：visit维度 * patients维度 * 预定义的dimension
    x = np.zeros((maxlen, n_samples, options['inputDimSize'])).astype(config.floatX)
    y = np.zeros((maxlen, n_samples, options['numClass'])).astype(config.floatX)
    mask = np.zeros((maxlen, n_samples)).astype(config.floatX)

    # 对于seqs与labels中的每一个患者的visit序列
    for idx, (seq, lseq) in enumerate(zip(seqs, labels)):
        # 对于visit序列中除去最后一个的所有visit
        # x[:, idx, :] 表示针对某个患者的输入，其形式为 最多visit次数*inputDimSize,这里遍历是遍历每一个visit
        # 这里seq的形式为visits*icd_codes， seqs[-1]表示不关注最后一次visit
        # 这个for就相当于为每一个visit制作恰当的binary的输入
        for xvec, subseq in zip(x[:, idx, :], seq[:-1]):
            # subseq是一个疾病的list
            # 将该次visit中所有出现的疾病标为1
            # print(len(xvec)) 输出均为4894
            xvec[subseq] = 1.

        # 对于visit序列中除去第一个的所有visit
        # y与x相同，只是最终label的类别数目不一样
        for yvec, subseq in zip(y[:, idx, :], lseq[1:]):
            # 同样是标记
            yvec[subseq] = 1.

        # 用mask矩阵来记录每个患者有多长得sequence
        mask[:lengths[idx], idx] = 1.

    lengths = np.array(lengths, dtype=config.floatX)

    # 总结一下，这里x的维度是 maxlen, n_samples=100, options['inputDimSize']=4894
    # y的维度是 maxlen, n_samples=100, options['numClass']=942
    # print(x.shape, y.shape)
    return x, y, mask, lengths


def calculate_cost(test_model, dataset, options):
    batchSize = options['batchSize']
    n_batches = int(np.ceil(float(len(dataset[0])) / float(batchSize)))
    costSum = 0.0
    dataCount = 0
    for index in xrange(n_batches):
        # 取一个batch的数据，形式为 patients*visit*icd_codes
        batchX = dataset[0][index * batchSize:(index + 1) * batchSize]
        batchY = dataset[1][index * batchSize:(index + 1) * batchSize]
        # pad后的数据为 maxVisitLength * patients * dimension(feature or label)
        x, y, mask, lengths = padMatrix(batchX, batchY, options)
        cost = test_model(x, y, mask, lengths)
        costSum += cost * len(batchX)
        dataCount += len(batchX)
    return costSum / dataCount


def calculate_cost_rhys(test_model, dataset, options):
    batchSize = options['batchSize']
    n_batches = int(np.ceil(float(len(dataset[0])) / float(batchSize)))
    costSum = 0.0
    dataCount = 0

    truePositive = 0.
    trueNegative = 0.

    for index in xrange(n_batches):
        # 取一个batch的数据，形式为 patients*visit*icd_codes
        batchX = dataset[0][index * batchSize:(index + 1) * batchSize]
        batchY = dataset[1][index * batchSize:(index + 1) * batchSize]
        # pad后的数据为 maxVisitLength * patients * dimension(feature or label)
        x, y, mask, lengths = padMatrix(batchX, batchY, options)
        cost, y_hat = test_model(x, y, mask, lengths)

        idx_sorted = np.argsort(-y_hat, axis=2)
        # print(idx_sorted[:, :, :20],)
        for visit_idx in range(len(y)):
            for patient_idx in range(len(y[0])):
                top_k = idx_sorted[visit_idx][patient_idx][:20]
                for label_idx in range(len(y[0][0])):
                    if y[visit_idx][patient_idx][label_idx] == 1:
                        if label_idx in top_k:
                            truePositive += 1.
                        else:
                            trueNegative += 1.

        costSum += cost * len(batchX)
        dataCount += len(batchX)

    top_k_accuracy = truePositive / (truePositive + trueNegative)

    return costSum / dataCount, top_k_accuracy


def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


def build_tree(treeFile):
    treeMap = pickle.load(open(treeFile, 'rb'))
    # 形成一个矩阵，每一行代表着第一个元素的祖先的序列
    ancestors = np.array(treeMap.values()).astype('int32')

    # ancSize记录为（祖先的个数+1）
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)

    # leaves也是一个数组，大小与ancestors相同，均为 疾病个数*（祖先个数+1）
    leaves = np.array(leaves).astype('int32')
    return leaves, ancestors


def train_GRAM(
        seqFile='seqFile.txt',
        labelFile='labelFile.txt',
        treeFile='tree.txt',
        embFile='embFile.txt',
        outFile='out.txt',
        inputDimSize=100,
        numAncestors=100,
        embDimSize=100,
        hiddenDimSize=200,
        attentionDimSize=200,
        max_epochs=100,
        L2=0.,
        numClass=26679,
        batchSize=100,
        dropoutRate=0.5,
        logEps=1e-8,
        verbose=True
):
    options = locals().copy()

    leavesList = []
    ancestorsList = []
    for i in range(5, 0, -1):
        # An ICD9 diagnosis code can have at most five ancestors (including the artificial root) when using CCS multi-level grouper.
        leaves, ancestors = build_tree(treeFile + '.level' + str(i) + '.pk')
        # 设置为全局变量
        sharedLeaves = theano.shared(leaves, name='leaves' + str(i))
        sharedAncestors = theano.shared(ancestors, name='ancestors' + str(i))
        leavesList.append(sharedLeaves)
        ancestorsList.append(sharedAncestors)

    print 'Building the model ... ',
    params = init_params(options)
    # theano的全局变量
    tparams = init_tparams(params)
    # 构造了一堆变量，以及这些变量之间的数学关系，从而构建了模型
    use_noise, x, y, mask, lengths, cost, cost_noreg, y_hat = build_model(tparams, leavesList, ancestorsList, options)
    # 将model中的运算定义为一个函数
    get_cost = theano.function(inputs=[x, y, mask, lengths], outputs=cost_noreg, name='get_cost')
    get_cost_rhys = theano.function(inputs=[x, y, mask, lengths], outputs=[cost_noreg, y_hat], name='get_cost_rhys')
    print 'done!!'

    print 'Constructing the optimizer ... ',
    grads = T.grad(cost, wrt=tparams.values())
    f_grad_shared, f_update = adadelta(tparams, grads, x, y, mask, lengths, cost)
    print 'done!!'

    print 'Loading data ... ',
    trainSet, validSet, testSet = load_data(seqFile, labelFile)
    # np.ceil：向上取整，表示一个epoch内有多少个batch
    n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
    print 'done!!'

    print 'Optimization start !!'
    bestTrainCost = 0.0
    bestValidCost = 100000.0
    bestTestCost = 0.0
    bestTrainAcc = 0.0
    bestValidAcc = 0.0
    bestTestAcc = 0.0
    epochDuration = 0.0
    bestEpoch = 0
    logFile = outFile + '.log'
    # xrange与range作用相同，但产生的是一个生成器
    for epoch in xrange(max_epochs):
        iteration = 0
        costVec = []
        startTime = time.time()
        # 按照batchSize的大小，随机遍历整个数据集
        for index in random.sample(range(n_batches), n_batches):
            use_noise.set_value(1.)
            # 0是指x， 1是指y， 按batchSize的大小取部分训练数据
            batchX = trainSet[0][index * batchSize:(index + 1) * batchSize]
            batchY = trainSet[1][index * batchSize:(index + 1) * batchSize]
            # 该函数体现了预测下一次visit的思想!
            # 这里x跟y是处理后的原输入，batchX的输入仍然是patients*visits*icd_codes的矩阵
            x, y, mask, lengths = padMatrix(batchX, batchY, options)
            # 更新后x与y的维度见padMatrix函数末尾
            # 根据输出来计算梯度，并更新参数
            costValue = f_grad_shared(x, y, mask, lengths)
            f_update()
            costVec.append(costValue)

            if iteration % 100 == 0 and verbose:
                buf = 'Epoch:%d, Iteration:%d/%d, Train_Cost:%f' % (epoch, iteration, n_batches, costValue)
                print buf
            iteration += 1
        duration = time.time() - startTime
        use_noise.set_value(0.)
        trainCost = np.mean(costVec)

        # validCost = calculate_cost(get_cost, validSet, options)
        # 这里把y_hat给导出来，然后跟y一起计算
        _, train_metric = calculate_cost_rhys(get_cost_rhys, trainSet, options)
        validCost, valid_metric = calculate_cost_rhys(get_cost_rhys, validSet, options)
        testCost, test_metric = calculate_cost_rhys(get_cost_rhys, testSet, options)
        buf = 'Epoch:%d, Duration:%f, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
            epoch, duration, trainCost, validCost, testCost)
        print buf
        print2file(buf, logFile)
        buf = 'Train_Acc:%f, Valid_Acc:%f, Test_Acc:%f' % (train_metric, valid_metric, test_metric)
        print buf
        print2file(buf, logFile)
        epochDuration += duration
        # 保存效果最好的模型
        if validCost < bestValidCost:
            bestValidCost = validCost
            bestTestCost = testCost
            bestTrainCost = trainCost
            bestEpoch = epoch
            bestTrainAcc = train_metric
            bestValidAcc = valid_metric
            bestTestAcc = test_metric
            tempParams = unzip(tparams)
            np.savez_compressed(outFile + '.' + str(epoch), **tempParams)
    buf = 'Best Epoch:%d, Avg_Duration:%f, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
        bestEpoch, epochDuration / max_epochs, bestTrainCost, bestValidCost, bestTestCost)
    print buf
    print2file(buf, logFile)
    buf = 'Train_Acc:%f, Valid_Acc:%f, Test_Acc:%f' % (bestTrainAcc, bestValidAcc, bestTestAcc)
    print buf
    print2file(buf, logFile)


def parse_arguments(parser):
    parser.add_argument('seq_file', type=str, metavar='<visit_file>',
                        help='The path to the Pickled file containing visit information of patients')
    parser.add_argument('label_file', type=str, metavar='<label_file>',
                        help='The path to the Pickled file containing label information of patients')
    parser.add_argument('tree_file', type=str, metavar='<tree_file>',
                        help='The path to the Pickled files containing the ancestor information of the input medical codes. Only use the prefix and exclude ".level#.pk".')
    parser.add_argument('out_file', metavar='<out_file>',
                        help='The path to the output models. The models will be saved after every epoch')
    parser.add_argument('--embed_file', type=str, default='',
                        help='The path to the Pickled file containing the representation vectors of medical codes. If you are not using medical code representations, do not use this option')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='The dimension size of the visit embedding. If you are providing your own medical code vectors, this value will be automatically decided. (default value: 128)')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='The dimension size of the hidden layer of the GRU (default value: 128)')
    parser.add_argument('--attention_size', type=int, default=128,
                        help='The dimension size of hidden layer of the MLP that generates the attention weights (default value: 128)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of training epochs (default value: 100)')
    parser.add_argument('--L2', type=float, default=0.001,
                        help='L2 regularization coefficient for all weights except RNN (default value: 0.001)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate used for the hidden layer of RNN (default value: 0.5)')
    parser.add_argument('--log_eps', type=float, default=1e-8,
                        help='A small value to prevent log(0) (default value: 1e-8)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print output after every 100 mini-batches (default false)')
    args = parser.parse_args()
    return args


def calculate_dimSize(seqFile):
    seqs = pickle.load(open(seqFile, 'rb'))
    codeSet = set()
    for patient in seqs:
        for visit in patient:
            for code in visit:
                codeSet.add(code)
    return max(codeSet) + 1


def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    rootCode = tree.values()[0][1]
    return rootCode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    # 计算特征的维度，即统计seq中出现的所有code的数量！！！
    inputDimSize = calculate_dimSize(args.seq_file)

    # 计算输出类别的维度，即统计label_file中出现的所有code的数量
    numClass = calculate_dimSize(args.label_file)

    # 获取rootCode的代码
    numAncestors = get_rootCode(args.tree_file + '.level2.pk') - inputDimSize + 1

    train_GRAM(
        seqFile=args.seq_file,
        inputDimSize=inputDimSize,
        treeFile=args.tree_file,
        numAncestors=numAncestors,
        labelFile=args.label_file,
        numClass=numClass,
        outFile=args.out_file,
        embFile=args.embed_file,
        embDimSize=args.embed_size,
        hiddenDimSize=args.rnn_size,
        attentionDimSize=args.attention_size,
        batchSize=args.batch_size,
        max_epochs=args.n_epochs,
        L2=args.L2,
        dropoutRate=args.dropout_rate,
        logEps=args.log_eps,
        verbose=args.verbose
    )
