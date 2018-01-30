import os
from keras import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, GRU
from preprocessor import load_data
from keras.utils import to_categorical

MODEL_FILE_NAME = 'model.h5'
CHECKPOINT_WEIGHTS_DIR = './cp_weights'
EPOCHS = 300
BATCH_SIZE = 8
NETWORK_SIZE = (2, 32)
DATA_SIZE = 100
CELL_TYPE = 'gru'

(x_train, y_train), VOC = load_data(limit=DATA_SIZE)
VOC_SIZE = len(VOC)

print('training set information:')
print('vocabulary size:', VOC_SIZE)
print('X shape:', x_train.shape)
print('Y shape:', y_train.shape)


def build_graph():
    model = Sequential()
    model.add(Embedding(VOC_SIZE, 10))
    for l in range(NETWORK_SIZE[0]):
        if CELL_TYPE == 'gru':
            model.add(GRU(NETWORK_SIZE[1], return_sequences=True, kernel_initializer='random_uniform'))
        elif CELL_TYPE == 'lstm':
            model.add(LSTM(NETWORK_SIZE[1], return_sequences=True, kernel_initializer='random_uniform'))
    model.add(TimeDistributed(Dense(VOC_SIZE, activation='softmax')))
    model.compile('rmsprop', 'categorical_crossentropy')
    return model


def train(model, x, y):
    # one-hot encode
    y = to_categorical(y, VOC_SIZE)

    try:
        if not os.path.exists(CHECKPOINT_WEIGHTS_DIR):
            os.mkdir(CHECKPOINT_WEIGHTS_DIR)

        # Keras callbacks
        checkpoint = ModelCheckpoint(CHECKPOINT_WEIGHTS_DIR + '/{epoch:d}.hdf5', save_weights_only=True, period=10)
        tensorboard = TensorBoard(batch_size=64)
        try:
            print('try restoring ...')
            epoch = 1
            files = os.listdir(CHECKPOINT_WEIGHTS_DIR)
            if len(files) > 0:
                # latest check points
                latest_cp = max(map(lambda f: int(f.replace('.hdf5', '')), files))
                epoch = latest_cp
                print('from epoch:', epoch)
            model.load_weights(CHECKPOINT_WEIGHTS_DIR + '/{}.hdf5'.format(epoch))
            model.fit(x, y, shuffle=True, batch_size=BATCH_SIZE, epochs=EPOCHS, initial_epoch=epoch,
                      callbacks=[checkpoint, tensorboard])
        except OSError:
            print('weights not found, continuing start new training')
            model.fit(x, y, batch_size=BATCH_SIZE, shuffle=True

                      , epochs=EPOCHS, callbacks=[checkpoint, tensorboard])
    except KeyboardInterrupt:
        print('')
        print('interrupted by user')
        pass


model = build_graph()
train(model, x_train, y_train)
model.save(MODEL_FILE_NAME)
