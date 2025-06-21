from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os 
no_sequences = 100 # Number of videos per action
sequence_length = 30 # Length of each video sequence: 30 frames
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Sign_language_action_recognition/data')

actions = np.array(['hello', 'thanks', 'iloveyou', 'yes', 'no', 'please', 'sorry','goodbye',
                    'stop', 'wait', 'go', 'come', 'love', 'help', 'like', 'dislike', 
                    'happy', 'sad', 'I', 'you', 'it'])

label_map = {label : num for num, label in enumerate(actions)}
#print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = 'logs'
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))  # 126 = 21*3*2
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=64) 

model.save('best_model.h5') 