import math
import requests
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import class_weight

'''
Currently:

NEED TO UPDATE TO NEW API
https://gitlab.com/dword4/nhlapi/-/blob/master/new-api.md

No normalization at the moment

Need to correct:
    Angle data and speed data to reflect + and - angles for both sides of the net

Need to add players info with feature hashing so can input into NN
Training not great right now, need to increase complexity of model maybe?
Add shift data?
'''


# Custom functions defined
def custom_standardize(input_string):
    """
    Preprocessing of strings before text vectorization

    :param input_string: string input to preprocess before text vectorization
    :return: lowercase of inp string
    """
    return tf.strings.lower(input_string)


def time_process(time: str):
    """
    Preprocess time into period into number of seconds into period as int

    :param time: string of time into period in the form mm:ss
    :return: number seconds into the period as int
    """
    mins, sec = time.split(":")
    seconds = int(mins) * 60 + int(sec)
    return seconds


def dist_angle(event):
    """
    Calculate and return distance and angle of input event if it has coordinate (x, y) data,
    else return -1's for everything

    :param event: event dictionary from JSON
    :return: list with 4 elements = [x, y, distance from net, angle to net]
    """
    # Get coordinates and calculate distance/angle and add to input tensor
    in_vect = [-1, -1, -1, -1]  # Problem if representing 0 angle and 0 distance if no angle or distance
    if "coordinates" in event:
        shot_deets = dict(event['coordinates'])
        if 'y' in shot_deets and 'x' in shot_deets:
            x = 100 - abs(shot_deets['x'])
            y = abs(shot_deets['y'])

            if x > 10:  # In front of goal line
                angle = 180 * math.atan(y / (x - 10)) / math.pi
            elif x == 0:
                return in_vect
            else:  # Behind goal line
                angle = 180 - 180 * math.atan(y / x) / math.pi
            in_vect = [x, y, int(math.sqrt((x - 10) ** 2 + y ** 2)), int(angle)]
    return in_vect


def get_shift_data(input_year_02=202202, num_games=1312, start=1):
    """

    STILL A WORK IN PROGRESS

    :param input_year_02:
    :param num_games:
    :param start:
    :return:
    """

    # Link to get shift data
    # https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId= {GAME ID HERE}


def get_shot_data(input_year_02=202202, num_games=1312, start=1):
    """
    Process all shot data from number games starting from start throughout reg season of given input year

    :param input_year_02: year + 02 for reg season
    :param num_games: number of games to process shot data for
    :param start: where in the season to start processing shot data
    :return: tuple of raw input shot data and raw labels for shot data
    """
    nn_in = []
    nn_out = []

    # Get JSON game data from NHL api for (default=every) game(s) in that season
    for game_num in range(start, start + num_games):
        game_id = str(input_year_02) + f"{game_num:04}"
        print(f"Getting stats from game {game_id}")
        game_url = "https://statsapi.web.nhl.com/api/v1/game/" + game_id + "/feed/live"
        response = requests.get(game_url, params={"Content-Type": "application/json"})
        data = response.json()
        response.close()
        if 'liveData' not in data:
            break
        # Produce clean distance and angle data for each shot along with labels of goal (1) or not (0)
        for shot_type in ("GOAL", "MISS", "MISSED_SHOT", "SHOT"):
            for shot in filter(lambda x: x["result"]["eventTypeId"] == shot_type,
                               data["liveData"]["plays"]["allPlays"]):
                # Get Previous Event
                shot_ind = shot['about']['eventIdx']
                prev_event = data["liveData"]["plays"]["allPlays"][shot_ind - 1]

                # 1 for goal, 0 for not goal
                typ = (0, 1)[shot_type == "GOAL"]
                nn_out.append(typ)

                # Add shot distance and angles
                processed = dist_angle(shot)
                processed.extend(dist_angle(prev_event))

                # Add time since last event and binary of previous event was a shot or shot attempt
                time_current = time_process(shot['about']['periodTime'])
                time_prev = time_process(prev_event['about']['periodTime'])
                time_diff = time_current - time_prev
                shot_or_not = int(
                    prev_event['result']['eventTypeId'] in ["SHOT", "MISS", "BLOCKED_SHOT", "MISSED_SHOT"])

                # Angular speed if both angles calculable, else -1
                # THIS IS INCORRECT AT THE MOMENT, TAKES ONLY POSITIVE ANGLES AND FIND DIFF
                if shot_or_not and time_diff != 0 and processed[3] > -1 and processed[7] > -1:
                    ang_speed = abs(processed[3] - processed[7]) // time_diff
                else:
                    ang_speed = -1

                # Linear speed
                distance_apart = round(
                    math.sqrt((processed[0] - processed[4]) ** 2 + (processed[1] - processed[5]) ** 2))
                if time_diff != 0 and processed[2] > -1 and processed[6] > -1:
                    lin_speed = distance_apart // time_diff
                else:
                    lin_speed = -1

                shot_add = [time_diff,  # Add time since previous event
                            shot_or_not,  # Add bool if was shot or shot attempt
                            ang_speed,
                            lin_speed]

                # Add change in angle from previous event if was shot
                processed.extend(shot_add)

                # Add binary for empty net or not
                if "emptyNet" in shot["result"]:
                    empt_net = (0, 1)[shot["result"]["emptyNet"] is True]
                else:
                    empt_net = 0
                processed.append(empt_net)

                # Add shot type as a string
                if "secondaryType" in shot["result"]:
                    processed.append(str(shot["result"]["secondaryType"]))
                else:
                    processed.append('UNK')

                # Previous event type
                if str(prev_event["result"]['eventTypeId']) == "MISSED_SHOT":
                    processed.append("MISS")
                else:
                    processed.append(prev_event['result']['eventTypeId'])

                # Add vector of data for that shot to input tensor
                nn_in.append(processed)

    nn_in[:][0] = np.array(nn_in[:][0])
    nn_out = np.array(nn_out)

    return nn_in, nn_out


def plot_loss(his, title="Training v Validation Loss"):
    """
    Plot the training loss and the validation loss over epochs

    :param his: history dictionary of model.fit
    :param title: title of plot
    :return: plt.show()
    """
    epochs_list = range(1, len(his.history['loss']) + 1)
    plt.plot(epochs_list, his.history['loss'], "b", linewidth=1, label="Training")
    plt.plot(epochs_list, his.history['val_loss'], "bo", markersize=1, label="Validation")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 5)
    plt.grid(True)
    plt.legend()
    plt.show()


def get_optimizer(learningrate_sched):
    """
    Return ADAM optimizer using given learning rate schedule

    :param learningrate_sched: learning rate schedule produced by tf.keras.optimizers.schedules.InverseTimeDecay()
    :return: ADAM optimizer from tf.keras.optimizers.Adam() using learningrate_sched
    """
    return tf.keras.optimizers.Adam(learningrate_sched)


def get_callbacks(patience=8, monitor='val_binary_crossentropy'):
    """
    Adding callbacks to training, particularly EarlyStoppage to prevent overfitting

    :param patience: epoch patience on early stoppage
    :param monitor: metric to monitor
    :return: list of callbacks, all from tf.keras.callbacks
    """
    return [
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)
    ]


def plot_lr(learnrate_sched, steps_pepoch=20):
    """
    Plot to how learning rate changes over epochs

    :param learnrate_sched: InverseTimeDecay schedule from tf.keras.optimizers.schedules
    :param steps_pepoch: steps per epoch (how many batches processed per epoch)
    :return: None
    """
    step = np.linspace(0, 1000)
    # Graph the learning rate decay
    lr = learnrate_sched(step)
    plt.figure(figsize=(8, 6))
    plt.plot(step / steps_pepoch, lr)
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.show()


def compile_and_fit(model, optimizer=None, max_epochs=1000, patience=8, monitor='val_binary_crossentropy'):
    """

    Compile and fit the given model

    :param model: Model using to compile and fit
    :param optimizer: optimizer being used (ADAM default)
    :param max_epochs: maximum number of epochs to train
    :param patience: epoch patience on EarlyStoppage callback
    :param monitor: metric to monitor for EarlyStoppage
    :return: history dictionary from model fitting
    """
    if optimizer is None:
        optimizer = get_optimizer(lr_sched)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[
                      tf.keras.metrics.BinaryCrossentropy(
                          from_logits=False, name='binary_crossentropy'), 'accuracy'])
    model.summary()

    history = model.fit(
        concat_input,
        train_label,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_split=0.2,
        class_weight=class_weight_dict,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(patience=patience, monitor=monitor),
        verbose=2)
    return history


def text_vect(train_ds, num_feat=13, seq_len=1, type_ev=12, type_shot=8, verbose=True):
    # TextVectorization layers
    # Vectorize shot types
    vect_shot_type = tf.keras.layers.TextVectorization(
        standardize=custom_standardize,
        max_tokens=type_shot,
        output_mode='int',  # To create unique integer indices for each token
        output_sequence_length=seq_len)

    # Vectorize event types
    vect_prev_type = tf.keras.layers.TextVectorization(
        standardize=custom_standardize,
        max_tokens=type_ev,
        output_mode='int',  # To create unique integer indices for each token
        output_sequence_length=seq_len)

    # Adapt, apply and onehot encode the vectorization layer
    vect_shot_type.adapt([row[num_feat] for row in train_ds])
    vect_prev_type.adapt([row[num_feat + 1] for row in train_ds])

    if verbose:  # Show what got vectorized
        for i in range(1, len(vect_shot_type.get_vocabulary())):
            print(f"Vectorized into {i} means--> {vect_shot_type.get_vocabulary()[i]}")
        for i in range(1, len(vect_prev_type.get_vocabulary())):
            print(f"Vectorized into {i} means--> {vect_prev_type.get_vocabulary()[i]}")

    return vect_prev_type, vect_shot_type

'''
These produce tensors of: 
    [Shot x, 
    shot y, 
    Shot distance, 
    shot angle, 
    prev event x,
    prev event y, 
    prev event distance, 
    prev event angle, 
    time since prev event, 
    prev event shot attempt?,
    angular speed from prev event, 
    linear speed from prev event, 
    empty net?, 
    shot type string (to be 8 one hot encoded),
    prev event type (to be 12 one hot encoded),
    ]
'''

# 100 000 for training with batches of 5000
N_TRAIN = int(1e5)
BATCH_SIZE = int(5e3)
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
NUM_TESTS = 100  # Number of random shots to test
rando = random.randint(27, 48908)

train_in_11, train_lab_11 = get_shot_data(201102, 1230)
train_in_12, train_lab_12 = get_shot_data(201202, 720)
train_in_13, train_lab_13 = get_shot_data(201302, 1230)
train_in_14, train_lab_14 = get_shot_data(201402, 1230)
train_in_15, train_lab_15 = get_shot_data(201502, 1230)
train_in_16, train_lab_16 = get_shot_data(201602, 1230)
train_in_17, train_lab_17 = get_shot_data(201702, 1271)
train_in_18, train_lab_18 = get_shot_data(201802, 1271)
train_in_19, train_lab_19 = get_shot_data(201902, 1082)
train_in_20, train_lab_20 = get_shot_data(202002, 868)
train_in_21, train_lab_21 = get_shot_data(202102, 1312)
train_in_22, train_lab_22 = get_shot_data(202202, 1312)

train_in, train_lab = list(train_in_11), list(train_lab_11)
for inp, lab in zip([train_in_12,
                     train_in_13,
                     train_in_14,
                     train_in_15,
                     train_in_16,
                     train_in_17,
                     train_in_18,
                     train_in_19,
                     train_in_20,
                     train_in_21],
                    [train_lab_12,
                     train_lab_13,
                     train_lab_14,
                     train_lab_15,
                     train_lab_16,
                     train_lab_17,
                     train_lab_18,
                     train_lab_19,
                     train_lab_20,
                     train_lab_21]):
    train_in.extend(inp)
    train_lab.extend(lab)

test_in, test_lab = train_in_22, train_lab_22

# # TEST TEST TEST TEST
# testy1, testy2 = get_shot_data(202202)
# testy_in, testy_lab = get_shot_data(202102)
# train_in, train_lab = testy1, testy2
# test_in, test_lab = testy_in, testy_lab

NUM_FEAT = 13
TYPE_SHOTS = 8
TYPE_EVENTS = 12
SEQ_LEN = 1
TOTAL_FEATS = NUM_FEAT + TYPE_SHOTS + TYPE_EVENTS


vect_prev, vect_shot = text_vect(train_in)
one_hot_vocab = tf.one_hot(vect_shot([row[NUM_FEAT] for row in train_in]), len(vect_shot.get_vocabulary()))
one_hot_prev = tf.one_hot(vect_prev([row[NUM_FEAT + 1] for row in train_in]), len(vect_prev.get_vocabulary()))
one_hot_test = tf.one_hot(vect_shot([row[NUM_FEAT] for row in test_in]), len(vect_shot.get_vocabulary()))
one_hot_prevtest = tf.one_hot(vect_prev([row[NUM_FEAT + 1] for row in test_in]), len(vect_prev.get_vocabulary()))
one_hot_vocab = tf.squeeze(one_hot_vocab, axis=1)
one_hot_prev = tf.squeeze(one_hot_prev, axis=1)
one_hot_prevtest = tf.squeeze(one_hot_prevtest, axis=1)
one_hot_test = tf.squeeze(one_hot_test, axis=1)

# Create tensors from raw input data that are numerical
numputs = tf.constant([row[:NUM_FEAT] for row in train_in])
numputs_test = tf.constant([row[:NUM_FEAT] for row in test_in])
binaries_in = tf.cast(numputs, tf.float32)
binaries_test = tf.cast(numputs_test, tf.float32)

# Create one big input tensor from one hot encodeds and numericals
concat_input = tf.concat([one_hot_vocab, binaries_in, one_hot_prev], axis=-1)
concat_test = tf.concat([one_hot_test, binaries_test, one_hot_prevtest], axis=-1)

# Gradually reduce learning rate during training (hyperbolically decrease lr)
lr_sched = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.01,
    decay_steps=5,
    decay_rate=0.02, staircase=False)

# Plot learning rate change over epochs
plot_lr(lr_sched)

# Create class weights dict that can account for imbalance of goals/no goals in fitting
class_weights = class_weight.compute_class_weight("balanced", classes=[0, 1], y=train_lab)
class_weight_dict = dict(enumerate(class_weights))

# Create Constant Tensors for ease of computing and leaving in/out tensors untouched
concat_input = tf.constant(concat_input)
train_label = tf.constant(train_lab)
concat_test = tf.constant(concat_test)
test_label = tf.constant(test_lab)

tiny_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='elu', input_shape=(TOTAL_FEATS,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
], name="Tiny")

medium_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(TOTAL_FEATS,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
], name="Medium")

med_model_tanh = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="tanh", input_shape=(TOTAL_FEATS,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation="tanh"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation="tanh"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
], name="Medium_tanh")

# Compile and Train models and store progress
fit_small = compile_and_fit(tiny_model)
fit_medium = compile_and_fit(medium_model)
fit_med_tanh = compile_and_fit(med_model_tanh)
size_histories = {'Small Dense with Dropouts': fit_small,
                  'Medium Dense with More Dropout': fit_medium,
                  'Medium Dense with tanh': fit_med_tanh}
plot_loss(fit_small, title="Small Model Training vs Validation")
plot_loss(fit_medium, title="Medium Model Training vs Validation")
plot_loss(fit_med_tanh, title="Medium TANH Training vs Validation")

# Evaluate models with new unseen test data
for mod in [tiny_model, medium_model, med_model_tanh]:
    loss, bin_entr, acc = mod.evaluate(concat_test, test_label, verbose=2)
    print(f"{mod.name} \n Evaluation loss, binary crossentropy, acc = {(loss, bin_entr, acc)}")

    # Prediction of some shots
    predictions = mod.predict(concat_test[rando:rando + NUM_TESTS])
    print("\n" + f"{mod.name} Predictions:")
    for pred, truth in zip(predictions, test_label[rando:rando + NUM_TESTS]):
        state = ('No Goal', 'Goal')[truth]
        print(f"Predicted: {pred[0] * 100:.2f}% vs Truth: {state}")
