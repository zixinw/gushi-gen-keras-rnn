import numpy as np
from collections import Counter

TOKEN_BEGIN = '^'
TOKEN_END = '$'
TOKEN_EMPTY = ' '


def process_poems(path_to_file, limit=1):
    poems = []
    total_content_len = 0
    with open(path_to_file, encoding='utf-8') as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                if len(title) > 10 or len(content) > 79:
                    continue

                content = ''.join([TOKEN_BEGIN, content, TOKEN_END])
                total_content_len += len(content)
                poems.append((title, content))
            except ValueError:
                pass

            limit -= 1
            if limit == 0:
                break

    # print(poems)
    # vocabulary = set()
    # vocabulary.add(TOKEN_EMPTY)
    vocabulary = []
    for title, content in poems:
        # vocabulary |= set([w for w in title])
        # vocabulary |= set([w for w in content])
        vocabulary += [w for w in title]
        vocabulary += [w for w in content]

    vocabulary.append(TOKEN_EMPTY)

    counter = Counter(vocabulary)
    vocabulary, vocabulary_counts = zip(*sorted(counter.items(), key=lambda x: -x[1]))
    vocabulary = dict(zip(vocabulary, range(len(vocabulary))))

    print('total content length of poems:', total_content_len)
    return vocabulary, poems


# def serialize(vocabulary, poems):
#     poem_vectors = []
#
#     total_vector_num = 0
#     # to vectors
#     for title, content in poems:
#         title = [vocabulary[w] for w in title]
#         content = [vocabulary[w] for w in content]
#         partial_content = []
#         for word in content:
#             if word == vocabulary[TOKEN_BEGIN]:
#                 partial_content.append(word)
#                 continue
#
#             poem_vectors.append((title, partial_content.copy(), word))
#             partial_content.append(word)
#             total_vector_num += 1
#
#     print('total vector number of poems:', total_vector_num)
#
#     max_title_length = 0
#     max_partial_content_length = 0
#     for title, partial_content, next_word in poem_vectors:
#         if len(title) > max_title_length:
#             max_title_length = len(title)
#         if len(partial_content) > max_partial_content_length:
#             max_partial_content_length = len(partial_content)
#
#     print('max title length:', max_title_length)
#     print('max partial content length:', max_partial_content_length)
#
#     index = 0
#     title_list = []
#     content_list = []
#     next_word_list = []
#     for title, partial_content, next_word in poem_vectors:
#         title_placeholder = [vocabulary[TOKEN_EMPTY] for i in range(max_title_length)]
#         title_placeholder[-len(title):] = title
#         title_list.append(title_placeholder)
#
#         partial_content_palceholder = [vocabulary[TOKEN_EMPTY] for i in range(max_partial_content_length)]
#         partial_content_palceholder[-len(partial_content):] = partial_content
#         content_list.append(partial_content_palceholder)
#
#         next_word_list.append(next_word)
#         index += 1
#
#     # print(serialized_poems)
#     # return np.array(serialized_poems)
#     return np.array(title_list), np.array(content_list), np.array(next_word_list)


def word2int(vocabulary, poems):
    poem_vectors = []
    for poem in poems:
        title, content = poem
        v_title = [vocabulary.get(t, TOKEN_EMPTY) for t in title]
        v_content = [vocabulary.get(t, TOKEN_EMPTY) for t in content]
        poem_vectors.append((v_title, v_content))

    return poem_vectors, vocabulary


def load_data(limit=1000):
    voc, poems = process_poems('./data/poems.txt', limit=limit)
    data, _ = word2int(voc, poems)
    data_size = len(data)

    content = list(map(lambda x: x[1], data))
    max_content_length = max(map(len, content))
    x_data = np.full((data_size, max_content_length), voc[TOKEN_EMPTY], np.int32)

    for row in range(data_size):
        x_data[row, :len(content[row])] = content[row]

    y_data = x_data.copy()
    y_data[:, :-1] = x_data[:, 1:]

    return (x_data, y_data), voc


def load_data_with_title(limit=1000):
    voc, poems = process_poems('./data/poems.txt', limit=limit)
    data, _ = word2int(voc, poems)
    data_size = len(data)

    title = list(map(lambda x: x[0], data))
    content = list(map(lambda x: x[1], data))
    max_title_length = max(map(len, title))
    max_content_length = max(map(len, content))

    x_title = np.full((data_size, max_title_length), voc[TOKEN_EMPTY], np.int32)
    x_content = np.full((data_size, max_content_length), voc[TOKEN_EMPTY], np.int32)

    for row in range(data_size):
        x_title[row, :len(title[row])] = title[row]
        x_content[row, :len(content[row])] = content[row]

    y_content = x_content.copy()
    y_content[:, :-1] = x_content[:, 1:]

    assert x_title.shape[0] == x_content.shape[0] == y_content.shape[0]
    assert x_title.shape[1] == max_title_length
    assert x_content.shape[1] == max_content_length == y_content.shape[1]

    return (x_title, x_content, y_content), voc


if __name__ == '__main__':
    (x_title, x_content, y_content), voc = load_data_with_title(100)
    print(x_title.shape)
    print(x_content.shape)
    print(y_content.shape)
