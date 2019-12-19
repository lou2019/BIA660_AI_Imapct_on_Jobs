import gensim
import codecs


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        f = codecs.open(self.filename, 'r', encoding='utf-8')  # 如果这东西在迭代里面, 他就不行!
        for line in f:
            yield line.split()

def main(domain):
    source = '/content/drive/My Drive/Attention-Based-Aspect-Extraction-master/preprocessed_data/%s/train.txt' % (domain) # 还能这样传入参数
    model_file = '/content/drive/My Drive/Attention-Based-Aspect-Extraction-master/preprocessed_data/%s/w2v_embedding' % (domain)
    sentences = MySentences(source) # 把句子拆成list
    model = gensim.models.Word2Vec(sentences, size=300, window=10, min_count=5, workers=4) # 初步训练embedding
    # model.save(model_file) # 过时保存方式
    # model.wv.save_word2vec_format(model_file) # 保存W2V文件 在预处理完的数据下面

print('Pre-training word embeddings ...')

# 这里需要取消注释, 才能生成W2V
main('restaurant')
# main('beer')
# main('laptops')
