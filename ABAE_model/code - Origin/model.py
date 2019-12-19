import logging
import os
import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from keras.constraints import MaxNorm

from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_model(args, maxlen, vocab):
    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = K.l2_normalize(weight_matrix, axis=-1) # K表示调用该函数的当前layer
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(w_n.shape[0].value))) # 自身矩阵的内积的平方根-自身特征值 = 越小, 越说明分量为0
        return args.ortho_reg * reg # 这东西越小越好, 因为能保证各个特征分的越开

    vocab_size = len(vocab)

    if args.emb_name: # 获取已经保存的embedding???
        from w2vEmbReader import W2VEmbReader as EmbReader
        emb_reader = EmbReader(os.path.join("..", "preprocessed_data", args.domain), args.emb_name)
        aspect_matrix = emb_reader.get_aspect_matrix(args.aspect_size)
        args.aspect_size = emb_reader.aspect_size
        args.emb_dim = emb_reader.emb_dim

    ##### Inputs #####
    sentence_input = Input(shape=(maxlen,), dtype='int32', name='sentence_input')
    neg_input = Input(shape=(args.neg_size, maxlen), dtype='int32', name='neg_input')

    ##### Construct word embedding layer #####
    word_emb = Embedding(vocab_size, args.emb_dim,
                         mask_zero=True, name='word_emb',
                         embeddings_constraint=MaxNorm(10))

    ##### Compute sentence representation ##### pre-processing 根据attention组合句子
    e_w = word_emb(sentence_input) # 将input转换为embedding
    y_s = Average()(e_w) # 默认求平均 layer
    att_weights = Attention(name='att_weights',
                            W_constraint=MaxNorm(10),
                            b_constraint=MaxNorm(10))([e_w, y_s]) # attention layer
    z_s = WeightedSum()([e_w, att_weights]) # encoding layer

    ##### Compute representations of negative instances #####  增加准确性的tricks
    e_neg = word_emb(neg_input)
    z_n = Average()(e_neg)

    ##### Reconstruction ##### 构建dense层, 希望能够decoding attention sentences的特征
    p_t = Dense(args.aspect_size)(z_s)
    p_t = Activation('softmax', name='p_t')(p_t) # softmax一下, nodes数量不改变, 数值被soft了一下
    r_s = WeightedAspectEmb(args.aspect_size, args.emb_dim, name='aspect_emb',
                            W_constraint=MaxNorm(10),
                            W_regularizer=ortho_reg)(p_t) # 标准化0-10区间, 且正则项为自定义的ortho_reg

    ##### Loss #####
    loss = MaxMargin(name='max_margin')([z_s, z_n, r_s]) # 自定义loss function??? 这是在做啥???
    model = Model(inputs=[sentence_input, neg_input], outputs=[loss]) # negative input是需要自己分开数据集的吗??

    ### Word embedding and aspect embedding initialization ######
    if args.emb_name:
        from w2vEmbReader import W2VEmbReader as EmbReader
        logger.info('Initializing word embedding matrix')
        embs = model.get_layer('word_emb').embeddings
        K.set_value(embs, emb_reader.get_emb_matrix_given_vocab(vocab, K.get_value(embs)))
        logger.info('Initializing aspect embedding matrix as centroid of kmean clusters') # 为何初始化要用到kmeans
        K.set_value(model.get_layer('aspect_emb').W, aspect_matrix) # r-s

    return model
