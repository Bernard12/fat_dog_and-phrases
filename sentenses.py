# -*- coding: utf-8 -*-

import vk
import re
import numpy as np


def download_data():
    session = vk.Session(
        access_token='5da85af35da85af35da85af3735dca07e055da85da85af30767f49c54d25843813b74e6'
    )
    vk_api = vk.API(session)
    res = vk_api.wall.get(owner_id=-101621324, v='5.92', count=700)
    posts = res['items']
    result = ""
    for post in posts:
        post_text = post['text']
        if has_group_ref(post_text) or has_url(post_text):
            continue
        # result.append(post_text)
        result += post_text
    # print(posts[10]['text'])
    return result


def has_group_ref(text):
    return len(re.findall(r'\[.*\|.*\]', text)) > 0


def has_url(text):
    result = re.findall(r'http[s]{0,1}', text)
    return len(result) > 0


def simple_stemming(text):
    result = re.sub(r'[.;!?,\-«»\)\(–]', u' ', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    result = emoji_pattern.sub(r'', result)
    result = re.sub(r'\s{2,}', ' ', result)
    return result.lower()


def encode_text(text):
    splited_text = text.split()
    tokens = sorted(list(set(splited_text)))
    word_index = dict((c, i) for i, c in enumerate(tokens))
    index_word = dict((i, c) for i, c in enumerate(tokens))
    return (splited_text, word_index, index_word)


def prepare_train_data(sentence_size=2):
    text = download_data()
    text = simple_stemming(text)
    splited_text, word_index, index_word = encode_text(text)
    examples_count = len(splited_text) - 1 - sentence_size
    shape = (examples_count, len(word_index))
    xs = np.zeros(shape)
    ys = np.zeros(shape)
    for i in range(0, examples_count):
        for j in range(0, sentence_size):
            xs[i, word_index[splited_text[i+j]]] = 1
        ys[i, word_index[splited_text[i+sentence_size]]] = 1
    return (xs, ys, word_index, index_word)


def generate_first_word_index(low=0, high=1, n=1):
    return np.random.randint(low=low, high=high, size=n)


def from_vec_to_word(vec, index_to_word):
    pos = np.argmax(vec)
    return index_to_word[pos]


def from_word_to_vec(word, word_to_index):
    shape = (1, len(word_to_index))
    result = np.zeros(shape)
    print(result.shape)
    result[0][word_to_index[word]] = 1
    return result


def sum_from_vecs(vec_list):
    shape = vec_list[0].shape
    result = np.zeros(shape)
    for vec in vec_list:
        result += vec
    return result


def generate_sentense(model, word_to_index, index_to_word,
                      n=10, sentense_size=2):
    size = len(word_to_index)
    start_index = generate_first_word_index(low=0, high=size, n=sentense_size)
    init_vec = []
    sentense = []

    for inds in xrange(0, sentense_size-1, 1):
        next_word = index_to_word[start_index[inds]]
        next_vec = from_word_to_vec(next_word, word_to_index)
        sentense.append(next_word)
        init_vec.append(next_vec)

    for i in xrange(0, n, 1):
        vec = sum_from_vecs(init_vec)
        next_vec = model.predict(vec)
        next_vec = np.reshape(next_vec, (1, size))
        next_word = from_vec_to_word(next_vec, index_to_word)
        sentense.append(next_word)
        init_vec.append(next_vec)
        init_vec = init_vec[1:]
    return sentense


# text = download_data()
# text = simple_stemming(text)
# enc = encode_text(text)
# xs, ys, a, b = prepare_train_data(10)
