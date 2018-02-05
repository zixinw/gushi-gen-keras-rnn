from keras.models import load_model
from preprocessor import load_data
import numpy as np

_, VOC = load_data(limit=100)
r_VOC = dict(zip(VOC.values(), VOC.keys()))

model = load_model('model_title_based.h5')

print(VOC)

title = '初晴落景      '
title_int = []
for t in title:
    i = VOC.get(t)
    if i is None:
        raise Exception('{} is not in VOC'.format(t))
    title_int.append(i)
title_int = np.array([title_int])
print(title_int)

content = '^'
c_index = 1

while content[-1] != '$' and len(content) < 74:
    c_content = content
    for i in range(74 - c_index):
        c_content += ' '

    p = [VOC[i] for i in c_content]
    x = np.array([p])
    p = model.predict([title_int, x])
    print(p[0].shape)
    index = np.argmax(p[-1], axis=1)
    print(index)
    next_word = r_VOC.get(index[c_index - 1])
    content += next_word
    # print(len(content))
    # print(next_word)

    c_index += 1

# print(r_VOC)
print(content)
