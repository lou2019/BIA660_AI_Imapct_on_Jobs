import codecs
import re
import operator

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
def is_number(token): # 判断数字??在哪里用
    return bool(num_regex.match(token))

def create_vocab(domain, maxlen=0, vocab_size=0): # 需要决定最大长度, 和vocab = feature space的大小. 同时将统计的资料保存到一个文档
    # assert domain in {'restaurant', 'beer'}
    source = '../preprocessed_data/' + domain + '/train.txt'

    total_words, unique_words = 0, 0
    word_freqs = {} # 字典!!
    top = 0

    fin = codecs.open(source, 'r', 'utf-8') # 对已经preprocessed的数据进行读取, 做EDA
    for line in fin:
        words = line.split()
        if maxlen > 0 and len(words) > maxlen:
            continue

        for w in words:
            if not is_number(w):
                try:
                    word_freqs[w] += 1 # 字典!!
                except KeyError:
                    unique_words += 1
                    word_freqs[w] = 1 # 字典!!
                total_words += 1

    print ('   %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True) # 啥意思???

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print ('  keep the top %i words' % vocab_size)

    # Write (vocab, frequence) to a txt file
    vocab_file = codecs.open('../preprocessed_data/%s/vocab' % domain, mode='w', encoding='utf8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word + '\t' + str(0) + '\n')
            continue
        vocab_file.write(word + '\t' + str(word_freqs[word]) + '\n')
    vocab_file.close()

    return vocab

 # 这岂不是每一个phase都会读取一个文件! 一个train, 一个test.
 # 并且对这两个文件进行特殊词频统计, hit rate统计.

def read_dataset(domain, phase, vocab, maxlen): # 统计最长长度+字典index对应关系
    # assert domain in {'restaurant', 'beer'} # 如果不是receptionist 就报错. 因为数据还没进行预处理.
    assert phase in {'train', 'test'} # 如果不是在这两个里面就会报错. 为了确保自己参数填写正确

    source = '../preprocessed_data/' + domain + '/' + phase + '.txt'
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.strip().split()
        if maxlen > 0 and len(words) > maxlen:
            words = words[:maxlen]
        if not len(words):
            continue

        indices = []
        for word in words:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)

    print('   <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return data_x, maxlen_x


def get_data(domain, vocab_size=0, maxlen=0):
    print('Reading data from ' + domain)
    print(' Creating vocab ...')
    vocab = create_vocab(domain, maxlen, vocab_size) # 先返回vocab,
    print(' Reading dataset ...')
    print('  train set')
    train_x, train_maxlen = read_dataset(domain, 'train', vocab, maxlen) # 然后再对train+test中的词汇进行vocab计数.
    print('  test set')
    test_x, test_maxlen = read_dataset(domain, 'test', vocab, maxlen) # 对train+test中的词汇进行词频展示
    maxlen = max(train_maxlen, test_maxlen)
    return vocab, train_x, test_x, maxlen


if __name__ == "__main__":
    vocab, train_x, test_x, maxlen = get_data('restaurant')
    print(len(train_x))
    print(len(test_x))
    print(maxlen)
