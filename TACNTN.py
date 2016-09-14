import cPickle
import numpy as np
import theano
from gensim.models.word2vec import Word2Vec
from process_ubuntu import WordVecs
from logistic_sgd import LogisticRegression
from CNN import QALeNetConvPoolLayer
from Classifier import BilinearLR, MLP, TensorClassifier
from Optimization import  Adam
import theano.tensor as T

def get_idx_from_sent(sent, word_idx_map, max_l=50, filter_h=3):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for i, word in enumerate(words):
        if i >= max_l: break
        if word in word_idx_map:
            x.append(word_idx_map[word])
        # else:
        #     x.append(1)
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

def train_cnn(datasets,
        U,
        TW,# pre-trained word embeddings
        filter_hs=[3],           # filter width
        hidden_units=[100,2],
        shuffle_batch=True,
        n_epochs=25,
        lam=0,
        batch_size=20,
        lr_decay = 0.95,          # for AdaDelta
        sqr_norm_lim=9):          # for optimization
    """
    return: a list of dicts of lists, each list contains (ansId, groundTruth, prediction) for a question
    """
    U = U.astype(dtype=theano.config.floatX)
    rng = np.random.RandomState(3435)
    img_h = (len(datasets[0][0])-3) / 2
    img_w = (int)(U.shape[1])
    lsize, rsize = img_h, img_h
    #filter_w = img_w
    filter_w = 100
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        #pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
        pool_sizes.append((img_h-filter_h+1,1))

    print pool_sizes

    parameters = [("image shape",img_h,img_w), ("filter shape",filter_shapes),
                  ("hidden_units",hidden_units), ("batch_size",batch_size),
                  ("lambda",lam), ("learn_decay",lr_decay),
                  ("sqr_norm_lim",sqr_norm_lim), ("shuffle_batch",shuffle_batch)]
    print parameters

    index = T.lscalar()
    lx = T.matrix('lx')
    rx = T.matrix('rx')
    y = T.ivector('y')
    t = T.ivector()
    t2 = T.ivector()

    TWords = theano.shared(value = TW)
    Words = theano.shared(value = U, name = "Words")

    session_topic = TWords[t]
    res_topic = TWords[t2]

    # session_topic = Words[T.cast(session_topic.flatten(),dtype="int32")].reshape((lx.shape[0],20,Words.shape[1]))
    # res_topic = Words[T.cast(res_topic.flatten(),dtype="int32")].reshape((lx.shape[0],20,Words.shape[1]))

    llayer0_input = Words[T.cast(lx.flatten(),dtype="int32")].reshape((lx.shape[0],lx.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch
    rlayer0_input = Words[T.cast(rx.flatten(),dtype="int32")].reshape((rx.shape[0],rx.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch

    train_set, dev_set, test_set = datasets[0], datasets[1], datasets[2]
    train_set_lx = theano.shared(np.asarray(train_set[:,:lsize],dtype=theano.config.floatX),borrow=True)
    train_set_rx = theano.shared(np.asarray(train_set[:,lsize:lsize+rsize],dtype=theano.config.floatX),borrow=True)
    train_set_y =theano.shared(np.asarray(train_set[:,-3],dtype="int32"),borrow=True)
    train_set_t =theano.shared(np.asarray(train_set[:,-2],dtype="int32"),borrow=True)
    train_set_t2 =theano.shared(np.asarray(train_set[:,-1],dtype="int32"),borrow=True)

    val_set_lx = theano.shared(np.asarray(dev_set[:,:lsize],dtype=theano.config.floatX),borrow=True)
    val_set_rx = theano.shared(np.asarray(dev_set[:,lsize:lsize+rsize],dtype=theano.config.floatX),borrow=True)

    val_set_y =theano.shared(np.asarray(dev_set[:,-3],dtype="int32"),borrow=True)
    val_set_t =theano.shared(np.asarray(dev_set[:,-2],dtype="int32"),borrow=True)
    val_set_t2 =theano.shared(np.asarray(dev_set[:,-1],dtype="int32"),borrow=True)

    llayer0_input = (llayer0_input ).dimshuffle(0,'x',1,2)
    rlayer0_input = (rlayer0_input).dimshuffle(0,'x',1,2)


    conv_layers = []        # layer number = filter number
    llayer1_inputs = []      # layer number = filter number
    rlayer1_inputs = []      # layer number = filter number
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]

        conv_layer = QALeNetConvPoolLayer(rng, linp=llayer0_input, rinp=rlayer0_input,
                                filter_shape=filter_shape, poolsize=pool_size)

        llayer1_input = conv_layer.loutput.flatten(2)
        rlayer1_input = conv_layer.routput.flatten(2)
        conv_layers.append(conv_layer)
        llayer1_inputs.append(llayer1_input)
        rlayer1_inputs.append(rlayer1_input)

    llayer1_input = T.concatenate(llayer1_inputs,1) # concatenate representations of different filters
    rlayer1_input = T.concatenate(rlayer1_inputs,1) # concatenate representations of different filters
    hidden_units[0] = feature_maps*len(filter_hs)

    scale = np.sqrt(6. / 150)
    topic_convert_matrix = theano.shared(value = np.random.uniform(size=(50,100),low=-scale, high=scale)
                                         .astype(theano.config.floatX),borrow=True)
    topic_convert_b =theano.shared(value=np.zeros((100,),dtype=theano.config.floatX),borrow=True)
    #T_matrix = T.tanh(T.dot(session_embedding,topic_convert_matrix)+topic_convert_b)

    T_matrix = T.dot(llayer1_input,topic_convert_matrix)+topic_convert_b
    T_matrix2 = T.dot(rlayer1_input,topic_convert_matrix)+topic_convert_b

    def weightedelement(seq):
        #a = theano.tensor.nnet.softmax(seq)
        return theano.tensor.nlinalg.diag(seq)


    weight_t = T.nnet.softmax(T.batched_dot(session_topic,T_matrix))
    weight_t2 = T.nnet.softmax(T.batched_dot(res_topic,T_matrix2))

    weight,_ = theano.scan(weightedelement,
                          sequences=weight_t)
    weight2,_ = theano.scan(weightedelement,
                          sequences=weight_t2)

    topic_vector = T.sum(T.batched_dot(session_topic.dimshuffle(0,2,1),weight),2)
    response_vector = T.sum(T.batched_dot(res_topic.dimshuffle(0,2,1),weight2),2)


    test = theano.function([index], weight, givens={
        lx: train_set_lx[index*batch_size:(index+1)*batch_size],
        rx: train_set_rx[index*batch_size:(index+1)*batch_size],
        y: train_set_y[index*batch_size:(index+1)*batch_size],
        t: train_set_t[index*batch_size:(index+1)*batch_size],
        t2: train_set_t2[index*batch_size:(index+1)*batch_size]
    },on_unused_input='ignore')
    print test(1).shape
    print test(1)


    tensordim = 8

    tmp1 = TensorClassifier(rng,50,50,dim_tensor=tensordim)
    tmp2 = TensorClassifier(rng,100,50,dim_tensor=tensordim)

    #topic_vector = _dropout_from_layer(rng,topic_vector,0.2)
    #response_vector = _dropout_from_layer(rng,response_vector,0.2)

    output_1 = tmp1(llayer1_input,rlayer1_input,batch_size=batch_size)
    output_2 = tmp2(response_vector,llayer1_input,batch_size=batch_size)
    output_3 = tmp2(topic_vector,rlayer1_input,batch_size=batch_size)


    classifier = LogisticRegression(T.concatenate([output_1,output_2,output_3],1)
                                    ,tensordim * 3,2,rng)
    #classifier = BilinearLR(llayer1_input, rlayer1_input, hidden_units[0], hidden_units[0])
    #cost = classifier.get_cost(y)
    cost = classifier.negative_log_likelihood(y)
    error = classifier.errors(y)
    opt = Adam()
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    params += [topic_convert_matrix,topic_convert_b]
    params += [Words]
    params += tmp1.params
    params += tmp2.params
    params += [TWords]
    print len(params)
    grad_updates = opt.Adam(cost=cost,params=params,lr = 0.001) #opt.sgd_updates_adadelta(params, cost, lr_decay, 1e-8, sqr_norm_lim)


    train_model = theano.function([index], cost,updates=grad_updates, givens={
        lx: train_set_lx[index*batch_size:(index+1)*batch_size],
        rx: train_set_rx[index*batch_size:(index+1)*batch_size],
        y: train_set_y[index*batch_size:(index+1)*batch_size],
        t: train_set_t[index*batch_size:(index+1)*batch_size],
        t2: train_set_t2[index*batch_size:(index+1)*batch_size]
    },on_unused_input='ignore')
    val_model = theano.function([index],[cost,error],givens={
        lx: val_set_lx[index*batch_size:(index+1)*batch_size],
        rx: val_set_rx[index*batch_size:(index+1)*batch_size],
        y: val_set_y[index*batch_size:(index+1)*batch_size],
        t: val_set_t[index*batch_size:(index+1)*batch_size],
        t2: val_set_t2[index*batch_size:(index+1)*batch_size]
    },on_unused_input='ignore')
    best_dev = 1.
    n_train_batches = datasets[0].shape[0]/batch_size
    for i in xrange(5):
        cost = 0
        total = 0.
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            #print train_model(minibatch_index)
            batch_cost = train_model(minibatch_index)
            total = total + 1
            cost = cost + batch_cost
            if total % 50 == 0:
                print total, cost/total, batch_cost
        cost = cost / n_train_batches
        print "echo %d loss %f" % (i,cost)

        cost=0
        errors = 0
        j = 0
        for minibatch_index in xrange(datasets[1].shape[0]/batch_size):
            tcost, terr = val_model(minibatch_index)
            cost += tcost
            errors += terr
            j = j+1
        cost = cost / j
        errors = errors / j
        if cost < best_dev:
            best_dev = cost
            save_params(params,'model\\TiebaFinal.bin')
        print  "echo %d dev_loss %f" % (i,cost)
        print  "echo %d dev_loss %f" % (i,errors)


def predict_cnn(datasets,
        U,
        TW,# pre-trained word embeddings
        filter_hs=[3],           # filter width
        hidden_units=[100,2],
        shuffle_batch=True,
        n_epochs=25,
        lam=0,
        batch_size=20,
        lr_decay = 0.95,          # for AdaDelta
        sqr_norm_lim=9):          # for optimization
    """
    return: a list of dicts of lists, each list contains (ansId, groundTruth, prediction) for a question
    """
    U = U.astype(dtype=theano.config.floatX)
    rng = np.random.RandomState(3435)
    img_h = (len(datasets[0][0])-3) / 2
    img_w = (int)(U.shape[1])
    lsize, rsize = img_h, img_h
    #filter_w = img_w
    filter_w = 100
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        #pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
        pool_sizes.append((img_h-filter_h+1,1))

    print pool_sizes

    parameters = [("image shape",img_h,img_w), ("filter shape",filter_shapes),
                  ("hidden_units",hidden_units), ("batch_size",batch_size),
                  ("lambda",lam), ("learn_decay",lr_decay),
                  ("sqr_norm_lim",sqr_norm_lim), ("shuffle_batch",shuffle_batch)]
    print parameters

    index = T.lscalar()
    lx = T.matrix('lx')
    rx = T.matrix('rx')
    y = T.ivector('y')
    t = T.ivector()
    t2 = T.ivector()

    TWords = theano.shared(value = TW)
    Words = theano.shared(value = U, name = "Words")

    session_topic = TWords[t]
    res_topic = TWords[t2]

    # session_topic = Words[T.cast(session_topic.flatten(),dtype="int32")].reshape((lx.shape[0],20,Words.shape[1]))
    # res_topic = Words[T.cast(res_topic.flatten(),dtype="int32")].reshape((lx.shape[0],20,Words.shape[1]))

    llayer0_input = Words[T.cast(lx.flatten(),dtype="int32")].reshape((lx.shape[0],lx.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch
    rlayer0_input = Words[T.cast(rx.flatten(),dtype="int32")].reshape((rx.shape[0],rx.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch

    train_set, dev_set, test_set = datasets[0], datasets[1], datasets[2]
    train_set_lx = theano.shared(np.asarray(train_set[:,:lsize],dtype=theano.config.floatX),borrow=True)
    train_set_rx = theano.shared(np.asarray(train_set[:,lsize:lsize+rsize],dtype=theano.config.floatX),borrow=True)
    train_set_y =theano.shared(np.asarray(train_set[:,-3],dtype="int32"),borrow=True)
    train_set_t =theano.shared(np.asarray(train_set[:,-2],dtype="int32"),borrow=True)
    train_set_t2 =theano.shared(np.asarray(train_set[:,-1],dtype="int32"),borrow=True)

    val_set_lx = theano.shared(np.asarray(dev_set[:,:lsize],dtype=theano.config.floatX),borrow=True)
    val_set_rx = theano.shared(np.asarray(dev_set[:,lsize:lsize+rsize],dtype=theano.config.floatX),borrow=True)
    val_set_y =theano.shared(np.asarray(dev_set[:,-3],dtype="int32"),borrow=True)
    val_set_t =theano.shared(np.asarray(dev_set[:,-2],dtype="int32"),borrow=True)
    val_set_t2 =theano.shared(np.asarray(dev_set[:,-1],dtype="int32"),borrow=True)

    llayer0_input = (llayer0_input ).dimshuffle(0,'x',1,2)
    rlayer0_input = (rlayer0_input).dimshuffle(0,'x',1,2)


    conv_layers = []        # layer number = filter number
    llayer1_inputs = []      # layer number = filter number
    rlayer1_inputs = []      # layer number = filter number
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]

        conv_layer = QALeNetConvPoolLayer(rng, linp=llayer0_input, rinp=rlayer0_input,
                                filter_shape=filter_shape, poolsize=pool_size)

        llayer1_input = conv_layer.loutput.flatten(2)
        rlayer1_input = conv_layer.routput.flatten(2)
        conv_layers.append(conv_layer)
        llayer1_inputs.append(llayer1_input)
        rlayer1_inputs.append(rlayer1_input)

    llayer1_input = T.concatenate(llayer1_inputs,1) # concatenate representations of different filters
    rlayer1_input = T.concatenate(rlayer1_inputs,1) # concatenate representations of different filters
    hidden_units[0] = feature_maps*len(filter_hs)

    scale = np.sqrt(6. / 150)
    topic_convert_matrix = theano.shared(value = np.random.uniform(size=(50,100),low=-scale, high=scale)
                                         .astype(theano.config.floatX),borrow=True)
    topic_convert_b =theano.shared(value=np.zeros((100,),dtype=theano.config.floatX),borrow=True)
    #T_matrix = T.tanh(T.dot(session_embedding,topic_convert_matrix)+topic_convert_b)

    T_matrix = T.dot(llayer1_input,topic_convert_matrix)+topic_convert_b
    T_matrix2 = T.dot(rlayer1_input,topic_convert_matrix)+topic_convert_b

    def weightedelement(seq):
        #a = theano.tensor.nnet.softmax(seq)
        return theano.tensor.nlinalg.diag(seq)


    weight_t = T.nnet.softmax(T.batched_dot(session_topic,T_matrix))
    weight_t2 = T.nnet.softmax(T.batched_dot(res_topic,T_matrix2))

    weight,_ = theano.scan(weightedelement,
                          sequences=weight_t)
    weight2,_ = theano.scan(weightedelement,
                          sequences=weight_t2)

    topic_vector = T.sum(T.batched_dot(session_topic.dimshuffle(0,2,1),weight),2)
    response_vector = T.sum(T.batched_dot(res_topic.dimshuffle(0,2,1),weight2),2)

    tensordim = 8

    tmp1 = TensorClassifier(rng,50,50,dim_tensor=tensordim)
    tmp2 = TensorClassifier(rng,100,50,dim_tensor=tensordim)

    output_1 = tmp1(llayer1_input,rlayer1_input,batch_size=batch_size)
    output_2 = tmp2(response_vector,llayer1_input,batch_size=batch_size)
    output_3 = tmp2(topic_vector,rlayer1_input,batch_size=batch_size)


    test = theano.function([index], weight2 , givens={
        lx: train_set_lx[index*batch_size:(index+1)*batch_size],
        rx: train_set_rx[index*batch_size:(index+1)*batch_size],
        y: train_set_y[index*batch_size:(index+1)*batch_size],
        t: train_set_t[index*batch_size:(index+1)*batch_size],
        t2: train_set_t2[index*batch_size:(index+1)*batch_size]
    },on_unused_input='ignore')


    classifier = LogisticRegression(T.concatenate([output_1,output_2,output_3],1)
                                    ,tensordim * 3,2,rng)

    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    params += [topic_convert_matrix,topic_convert_b]
    params += [Words]
    params += tmp1.params
    params += tmp2.params
    params += [TWords]
    print len(params)

    load_params(params,r'model\backup\TiebaFinal.bin')
    cost = classifier.negative_log_likelihood(y)
    error = classifier.errors(y)
    predict = classifier.predict_prob

    val_model = theano.function([index],[y,predict,error,cost],givens={
        lx: val_set_lx[index*batch_size:(index+1)*batch_size],
        rx: val_set_rx[index*batch_size:(index+1)*batch_size],
        y: val_set_y[index*batch_size:(index+1)*batch_size],
        t: val_set_t[index*batch_size:(index+1)*batch_size],
        t2: val_set_t2[index*batch_size:(index+1)*batch_size]
    },on_unused_input='ignore')
    #print test(1).shape
    sdf = test(1)[0]
    print sdf.shape
    for i in range(20):
        print sdf[i][i]

    best_dev = 1.
    f = open('res.txt','w')
    n_train_batches = datasets[0].shape[0]/batch_size

    cost=0
    errors = 0
    j = 0
    for minibatch_index in xrange(datasets[1].shape[0]/batch_size):
        a,b ,terr,tcost = val_model(minibatch_index)
        cost += tcost
        errors += terr
        j = j+1

        for i in range(200):
            f.write(str(b[i][1]))
            f.write('\t')
            f.write(str(a[i]))
            f.write('\n')


    cost = cost / j
    errors = errors / j
    if cost < best_dev:
        best_dev = cost
    print  "echo %d dev_loss %f" % (0,cost)
    print  "echo %d dev_loss %f" % (0,errors)

def load_params(params,filename):
    f = open(filename)
    num_params = cPickle.load(f)
    for p,w in zip(params,num_params):
        p.set_value(w,borrow=True)
    print "load successfully"

def save_params(params,filename):
    num_params = [p.get_value() for p in params]
    f = open(filename,'wb')
    cPickle.dump(num_params,f)

def ComputeSame(m,r):
    total = 0.
    for term in m.split():
        if term in r:
            total = total + 1
    #print total
    return total

def make_cnn_data(revs, word_idx_map, max_l=50, filter_h=3, val_test_splits=[2,3]):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    val_split, test_split = val_test_splits
    for rev in revs:
        sent = get_idx_from_sent(rev["m"], word_idx_map, max_l, filter_h)
        sent += get_idx_from_sent(rev["r"], word_idx_map, max_l, filter_h)
        sent.append(int(rev["y"]))
        sent.append(int(rev["t"]))
        sent.append(int(rev["t2"]))

        if len(val) > 50000:
            train.append(sent)
        else:
            val.append(sent)

        if len(train) > 1200000:
            break

    if len(train) == 0:
        train = val
    train = np.array(train,dtype="int")
    val = np.array(val,dtype="int")
    test = np.array(test,dtype="int")
    print 'trainning data', len(train),'val data', len(val)
    return [train, val, test]

def createtopicvec(word_idx_map):
    max_topicword = 20
    topicmatrix = np.zeros((100,max_topicword)).astype('int')
    file = open('mergedic.txt')
    i = 0
    miss = 0
    for line in file:
        tmp = line.strip().split(' ')
        for j in range(min(len(tmp),max_topicword)):
            if tmp[j] in word_idx_map:
                topicmatrix[i,j] = word_idx_map[tmp[j]]
            else:
                topicmatrix[i,j] = 0
        i = i + 1
    return topicmatrix


if __name__=="__main__":
    dataset = r"data\raw_data4.bin"
    x = cPickle.load(open(dataset,"rb"))
    revs, wordvecs, max_l,tw = x[0], x[1], x[2], x[3]
    # datasets = make_cnn_data(revs,wordvecs.word_idx_map,max_l=20)
    # tm = createtopicvec(wordvecs.word_idx_map)
    #x = cPickle.load(open(r"data\testdata.bin","rb"))
    #revs, wordvecs2, max_l2 = x[0], x[1], x[2]
    datasets = make_cnn_data(revs,wordvecs.word_idx_map,max_l=20)
    train_cnn(datasets,wordvecs.W,tw,filter_hs=[3],hidden_units=[50],batch_size=200)
    predict_cnn(datasets,wordvecs.W,tw,filter_hs=[3],hidden_units=[50],batch_size=200)
