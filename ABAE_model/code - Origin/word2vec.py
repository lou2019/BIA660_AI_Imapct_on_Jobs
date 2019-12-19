import gensim
import codecs


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(domain):
    source = '../preprocessed_data/%s/train.txt' % (domain) # 还能这样传入参数
    model_file = '../preprocessed_data/%s/w2v_embedding' % (domain)
    sentences = MySentences(source) # 把句子拆成list
    model = gensim.models.Word2Vec(sentences, size=200, window=10, min_count=5, workers=4) # 初步训练embedding
    model.save(model_file) # 保存W2V文件 在预处理完的数据下面


print('Pre-training word embeddings ...')
# main('restaurant')
# main('beer')
# main('laptops')
