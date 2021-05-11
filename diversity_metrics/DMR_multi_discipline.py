import collections
import itertools
import tomotopy as tp
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
import string
from diversity_metrics import *

def unique_discipline_list():
    unique_list = []
    filename = r"Fields.txt"
    with open(filename, encoding='utf-8') as fp:
        for cnt, line in enumerate(fp):
            line = line.strip()
            if line not in unique_list:
                unique_list.append(line)
                #print(unique_list)

    return unique_list

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

trans_table = {ord(c): None for c in string.punctuation + string.digits}
stemmer = PorterStemmer()

def tokenize(text):
    # my text was unicode so I had to use the unicode-specific translate function. If your documents are strings, you will need to use a different `translate` function here. `Translated` here just does search-replace. See the trans_table: any matching character in the set is replaced with `None`
    tokens = [word for word in nltk.word_tokenize(text.translate(trans_table)) if len(
        word) > 1]  # if len(word) > 1 because I only want to retain words that are at least two characters before stemming, although I can't think of any such words that are not also stopwords
    stems = [stemmer.stem(item) for item in tokens]
    return stems

def cos_similarity(X):
    return (X * X.T).toarray()

def gini(array):
   array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

def get_data():
    filename = r"id,date,Dis,keywords.txt"
    id_term = {}
    id_year = {}
    id_discipline = {}
    with open(filename, encoding='utf-8') as fp:
        for cnt, line in enumerate(fp):
            fields = line.split('\t')
            id = fields[0]
            year = fields[1].split("-")[0]
            discipline = fields[2]
            terms = fields[3]
            id_term[id] = terms.replace(';', '')
            id_year[id] = year
            id_discipline[id] = discipline.split(';')

    return id_term, id_year, id_discipline

def train_model():
    documents = []
    id_term, id_year, id_discipline = get_data()
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    stemmer = PorterStemmer()
    stops = set(stopwords.words('english'))
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer())

    for i, key in enumerate(id_term):
        if len(id_year[key]) <= 4:
            #model.add_doc(id_term[key].split(), metadata=json.dumps({'year':id_year[key], 'uid':key}))
            corpus.add_doc(raw=id_term[key], uid=key, metadata=id_year[key])
            #model.add_doc(id_term[key].split(), metadata=id_year[key])
            if i % 10 == 0:
                print('Document #{} has been loaded'.format(i) + " : " + id_year[key])

    model = tp.DMRModel(k=15, alpha=0.1, eta=0.01, min_cf=5, rm_top=50, tw=tp.TermWeight.IDF, corpus=corpus)
    model.optim_interval = 20
    model.burn_in = 200
    model.train(0)

    print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
        len(model.docs), len(model.used_vocabs), model.num_words
    ))

    # Let's train the model
    for i in range(0, 2000, 20):
        print('Iteration: {:04} LL per word: {:.4}'.format(i, model.ll_per_word))
        model.train(20)
    print('Iteration: {:04} LL per word: {:.4}'.format(1000, model.ll_per_word))
    for k in range(model.k):
        print('Topic #{}'.format(k))
        for word, prob in model.get_topic_words(k, top_n=15):
            print('\t', word, prob, sep='\t')

    print('removed:' + str(model.removed_top_words))
    p_file = open(str(model.k) + "_perplexity.txt", "w")
    p_file.write(str(model.perplexity))
    print(model.perplexity)
    p_file.close()

    model.save("multi_disciplinarity_dmr.model")

    # calculate topic distribution for each metadata using softmax
    probs = np.exp(model.lambdas - model.lambdas.max(axis=0))
    probs /= probs.sum(axis=0)

    print('Topic distributions for each metadata')
    file = open("year_topic_distribution.txt", "w")
    for f, metadata_name in enumerate(model.metadata_dict):
        met = json.loads(metadata_name)
        #print(met)
        print(str(met), probs[:, f], '\n')
        file.write(str(met) + "\t" + str(probs[:, f]) + "\n")
    file.close()

    x = np.arange(model.k)
    width = 1 / (model.f + 2)

    fig, ax = plt.subplots()
    for f, metadata_name in enumerate(model.metadata_dict):
        met = json.loads(metadata_name)
        ax.bar(x + width * (f - model.f / 2), probs[:, f], width, label=met)

    ax.set_ylabel('Probabilities')
    ax.set_yscale('log')
    ax.set_title('Topic distributions')
    ax.set_xticks(x)
    ax.set_xticklabels(['Topic #{}'.format(k) for k in range(model.k)])
    ax.legend()

    fig.tight_layout()
    plt.show()

train = True
if train:
    train_model()
else:
    id_term, id_year, id_discipline = get_data()

    #unique discipline list
    unique_list = unique_discipline_list()

    #import gensim
    # load fasttext English embeddings
    #wv = gensim.models.fasttext.load_facebook_model('cc.en.300.bin.gz')

    topic_discipline = collections.defaultdict(list)
    #print(topic_discipline)
    model = tp.DMRModel.load('multi_disciplinarity_dmr.model')

    calculate topic distribution for each metadata using softmax
    probs = np.exp(model.lambdas - model.lambdas.max(axis=0))
    probs /= probs.sum(axis=0)
    print('Topic distributions for each metadata')
    file = open("year_topic_distribution.txt", "w")
    for f, metadata_name in enumerate(model.metadata_dict):
       met = json.loads(metadata_name)
       print(str(met))
       file.write(str(met) + "\t" + str(probs[:, f]) + "\n")
    file.close()
    ids = list(id_discipline.keys())
    id_list = {}
    for i, id in enumerate(ids):
        id_list[i] = id
    
    docs = model.docs
    for i, doc in enumerate(docs):
        meta = json.loads(doc.metadata)
        #print(str(meta) + " : " + str(doc.uid))
        id = doc.uid
        #id = id_list[i]
        t_docs = doc.get_topics()
        #print(t_docs)
        for i, t_doc in enumerate(t_docs):
            if t_doc[1] > 0.1 and i == 0:
                for dis in id_discipline[id]:
                    if dis in unique_list:
                        topic_discipline[t_doc[0]].append(dis)
    
    vectorizer1 = TfidfVectorizer(tokenizer=tokenize, binary=True, min_df=1, stop_words="english")
    topic_id_map = {}
    list_str = []
    print("variety")
    for i, key in enumerate(topic_discipline):
        doc = topic_discipline[key]
        listToStr = ' '.join([str(elem.replace('_','')) for elem in doc])
        list_str.append(listToStr)
        topic_id_map[i] = key
    
        #computing variety
        #unique_list = set(doc)
        #print(str(key) + " : " + str(len(unique_list)))
    
    X = vectorizer1.fit_transform(list_str)
    
    print("gini coefficient")
    #computing gini coefficient
    for i, a_array in enumerate(X.toarray()):
        gini_score = gini(a_array)
        print(str(topic_id_map[i]) + " : " + str(gini_score))
    
    #computing disparity
    print("disparity")
    cosine_mat = cos_similarity(X)
    for i, mat in enumerate(cosine_mat):
        avg_sim = 0.0
        for ele in mat:
            avg_sim += ele
        print(str(topic_id_map[i]) + " : " + str(avg_sim/len(mat)) + " : " + str(1-(avg_sim/len(mat))))

    print("topic diversity")
    n = 50
    print(len(topic_discipline))
    for t in topic_discipline:
        topics = topic_discipline[t]
        composite_list = list(divide_chunks(topics, n))
        print(len(topics))
        print(len(composite_list))
        final_list = []
        for a in composite_list:
            n_sent = ' '.join(a)
            n_sent = n_sent.replace(',', ' ')
            n_sent = n_sent.replace('_', ' ')
            final_list.append(tokenize(n_sent))
    
        print(str(t) + " : " + str(irbo(final_list, weight=0.98, topk=50)))
