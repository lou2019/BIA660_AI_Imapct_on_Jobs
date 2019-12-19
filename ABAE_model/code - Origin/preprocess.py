from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import json
from tqdm import tqdm
import argparse

# input: 在datasets文件夹下面, 读取domain的3个文件
    # f = codecs.open('../datasets/' + domain + '/train.txt', 'r', 'utf-8')
    # f1 = codecs.open('../datasets/' + domain + '/test.txt', 'r', 'utf-8')
    # f2 = codecs.open('../datasets/' + domain + '/test_label.txt', 'r', 'utf-8')

# output:返回已经处理完的txt文件, 每一行都是一个sentence的tokens.(包含3个文件,
    # out = codecs.open('../preprocessed_data/' + domain + '/train.txt', 'w', 'utf-8')
    # out1 = codecs.open('../preprocessed_data/' + domain + '/test.txt', 'w', 'utf-8')
    # out2 = codecs.open('../preprocessed_data/' + domain + '/test_label.txt', 'w', 'utf-8')

def parseSentence(line): # 以行为单位, 进行parsing
    lmtzr = WordNetLemmatizer() # 使用 Wordnet的 Lemmatizer
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop] # 这里有 Lemmatization
    return text_stem


def preprocess_train(domain): # 读取数据 for training
    f = codecs.open('../datasets/' + domain + '/train.txt', 'r', 'utf-8')
    out = codecs.open('../preprocessed_data/' + domain + '/train.txt', 'w', 'utf-8')

    for line in f:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out.write(' '.join(tokens) + '\n')


def preprocess_test(domain): # 这里是需要修改的部分, 我们没有dataset
    # For restaurant domain, only keep sentences with single 
    # aspect label that in {Food, Staff, Ambience}

    f1 = codecs.open('../datasets/' + domain + '/test.txt', 'r', 'utf-8')
    f2 = codecs.open('../datasets/' + domain + '/test_label.txt', 'r', 'utf-8')
    out1 = codecs.open('../preprocessed_data/' + domain + '/test.txt', 'w', 'utf-8')
    out2 = codecs.open('../preprocessed_data/' + domain + '/test_label.txt', 'w', 'utf-8')

    for text, label in zip(f1, f2):
        label = label.strip()
        if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']: # label没包含这些的删除. 而我们的dataset默认已经都是在receptionist下
            continue
        tokens = parseSentence(text)
        if len(tokens) > 0:
            out1.write(' '.join(tokens) + '\n')
            out2.write(label + '\n')


def preprocess_line(line):
    return " ".join([morph.parse(w)[0].normal_form for w in word_tokenize(line.lower())])


def preprocess_reviews_train(): # review 不一定会运行. 除非特定的名字叫app_reviews
    with open("../preprocessed_data/app_reviews/appstore.json", "rt") as f:
        reviews = json.load(f)
    with open("../preprocessed_data/app_reviews/train.txt", "wt") as f:
        for rev in tqdm(reviews):
            if isinstance(rev, dict):
                f.write(preprocess_line(rev["Title"] + " " + rev["Review"]) + "\n")


def preprocess(domain):
    print('\t' + domain + ' train set ...')
    preprocess_train(domain)
    print('\t' + domain + ' test set ...')
    preprocess_test(domain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant',
                        help="domain of the corpus")
    args = parser.parse_args()
    # review 不一定会运行. 除非特定的名字叫app_reviews
    if args.domain == "app_reviews":  # 如果是app_reviews 则将reviews的json格式转换为train_set, 如果不是的话, 这里不会运行
        import pymorphy2 # POS + 词性标注 + 词形变化引擎
        from nltk.tokenize import word_tokenize

        morph = pymorphy2.MorphAnalyzer()

        print('Preprocessing raw review sentences ...')
        preprocess_reviews_train()
    else:  # 将该domain的词汇进行 preprocess: 包括train+test
        preprocess(args.domain)
