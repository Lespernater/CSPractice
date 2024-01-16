"""
Currently:

No normalization at the moment
Need to correct angle data and speed data to reflect + and - angles for both sides of the net
Training not great right now, need to increase complexity of model maybe?
Add shift data?
Add teams (and who is home)

Still Broken:
Plotting in Pycharm
"""

import math
import requests
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import class_weight


class Event:
    def __init__(self, dict_from_api, prev_event=None):
        self.whole = dict_from_api
        self.clock = self.sec_remain()
        self.typev = self.whole.get('typeDescKey', "NULL")
        self.period_dic = self.whole.get('periodDescriptor', None)
        # self.home_sided = self.whole['homeTeamDefendingSide']
        self.details, self.team_owner = self.update_details()
        self.shot = int(self.update_shot())
        self.play1, self.play2 = self.update_players()
        self.loc_vect = self.locate()
        self.prev_event = prev_event
        self.time_since_prev = self.time_since_prev()
        self.shot_method = self.method()
        self.situation = self.whole.get('situationCode', "NULL")
        self.goal = int(self.typev == 'goal')

    def get_clock(self):
        return self.clock

    def get_loc(self):
        return self.loc_vect

    def get_type(self):
        return self.typev

    def get_goal(self):
        return self.goal

    def method(self):
        if self.details:
            return self.details.get('shotType', 'NULL')
        return "NULL"

    def locate(self):
        in_vect = [-1, -1, -1, -1, "NA"]
        if self.details:
            if "xCoord" in self.details and "yCoord" in self.details:
                in_vect[4] = self.whole['details']['zoneCode']
                x = 100 - abs(self.whole['details']['xCoord'])
                y = abs(self.whole['details']['yCoord'])
                if x > 10:  # In front of goal line
                    angle = 180 * math.atan(y / (x - 10)) / math.pi
                elif x == 0:
                    return in_vect
                else:  # Behind goal line
                    angle = 180 - 180 * math.atan(y / x) / math.pi
                in_vect = [int(x), int(y), int(math.sqrt((x - 10) ** 2 + y ** 2)), int(angle), in_vect[4]]
        return in_vect

    def outclean(self):
        if self.prev_event:
            outputs = combine_elements(self.loc_vect, self.prev_event.get_loc(), self.speed(),
                                       self.prev_event.time_since_prev, self.clock, self.typev, self.prev_event.typev,
                                       self.play1, self.play2, self.shot_method, self.team_owner, self.situation)
        else:
            outputs = combine_elements(self.loc_vect, [-1, -1, -1, -1, "NA"], self.speed(), -1, self.clock,
                                       self.typev, "NULL", self.play1, self.play2, self.shot_method,
                                       self.team_owner, self.situation)
        return outputs

    def sec_remain(self):
        mins, sec = str(self.whole['timeRemaining']).split(":")
        seconds = int(mins) * 60 + int(sec)
        return seconds

    def speed(self):
        ang_speed, lin_speed = -1, -1
        if self.prev_event:
            if self.loc_vect[0] > -1 and self.time_since_prev > 0 and self.loc_vect[3] > -1 and \
                    self.prev_event.loc_vect[3] > -1:
                ang_speed = int(abs(self.loc_vect[3] - self.prev_event.loc_vect[3]) // self.time_since_prev)
            if self.time_since_prev > 0 and self.loc_vect[2] > -1 and self.prev_event.loc_vect[2] > -1:
                distance_apart = int(round(
                    math.sqrt((self.loc_vect[0] - self.prev_event.loc_vect[0]) ** 2 + (
                            self.loc_vect[0] - self.prev_event.loc_vect[0]) ** 2)))
                lin_speed = distance_apart // self.time_since_prev
        return ang_speed, lin_speed

    def time_since_prev(self):
        if self.prev_event:
            return self.clock - self.prev_event.get_clock()
        return -1

    def update_details(self):
        out = [None, "NULL"]
        if 'details' in self.whole:
            out[0] = self.whole['details']
            if 'eventOwnerTeamId' in out[0]:
                out[1] = out[0]['eventOwnerTeamId']
        return out[0], out[1]

    def update_players(self):
        if self.details:
            if self.typev in ('takeaway', 'giveaway'):
                return self.details['playerId'], -1
            elif self.typev == 'hit':
                return self.details['hittingPlayerId'], self.details['hitteePlayerId']
            elif self.typev == 'blocked-shot':
                return self.details['blockingPlayerId'], self.details['shootingPlayerId']
            elif self.typev == 'faceoff':
                return self.details['winningPlayerId'], self.details['losingPlayerId']
            elif self.typev in ('missed-shot', 'shot-on-goal'):
                return self.details['shootingPlayerId'], self.details.get('goalieInNetId', -1)
            elif self.typev == 'goal':
                return self.details['scoringPlayerId'], self.details.get('goalieInNetId', -1)
        return -1, -1

    def update_shot(self):
        return self.typev in ("blocked-shot", "missed-shot", "goal", "miss", "shot-on-goal")


def combine_elements(*args):
    ints, non_ints = [], []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, int):
                    ints.append(item)
                else:
                    non_ints.append(item)
        elif isinstance(arg, int):
            ints.append(arg)
        else:
            non_ints.append(arg)
    combined_list = ints + non_ints
    return combined_list


# Custom functions defined
def time_process(time: str):
    """
    Preprocess time into period_dic into number of seconds into period_dic as int

    :param time: string of time into period_dic in the form mm:ss
    :return: number seconds into the period_dic as int
    """
    mins, sec = time.split(":")
    seconds = int(mins) * 60 + int(sec)
    return seconds


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
        game_url = "https://api-web.nhle.com/v1/gamecenter/" + game_id + "/play-by-play"
        response = requests.get(game_url, params={"Content-Type": "application/json"})
        data = response.json()
        response.close()
        if 'plays' not in data:
            break

        previous = None
        for i in range(len(data["plays"])):
            event = Event(data["plays"][i], previous)
            if event.get_type() in ("goal", "miss", "missed-shot", "shot-on-goal", "blocked_shot"):
                nn_out.append(event.get_goal())
                # Add vector of data for that shot to input tensor
                nn_in.append(event.outclean())
            previous = event

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
    if learningrate_sched:
        return tf.keras.optimizers.Adam(learningrate_sched)
    else:
        return None


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


def compile_and_fit(model, concat_input, train_label, optimizer=None,
                    max_epochs=1000, patience=20, monitor='val_binary_crossentropy'):
    """

    Compile and fit the given model

    :param concat_input:
    :param train_label:
    :param model: Model using to compile and fit
    :param optimizer: optimizer being used (ADAM default)
    :param max_epochs: maximum number of epochs to train
    :param patience: epoch patience on EarlyStoppage callback
    :param monitor: metric to monitor for EarlyStoppage
    :return: history dictionary from model fitting
    """
    n_train = int(1e5)
    batch_size = int(5e3)
    # Gradually reduce learning during training (hyperbolically decrease lr)
    lr_sched = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.01,
        decay_steps=5,
        decay_rate=0.02, staircase=False)

    # Plot learning rate change over epochs
    # plot_lr(lr_sched)

    if optimizer is None:
        optimizer = get_optimizer(lr_sched)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[
                      tf.keras.metrics.BinaryCrossentropy(
                          from_logits=False, name='binary_crossentropy'), 'accuracy'])
    model.summary()

    # Create class weights dict that can account for imbalance of goals/no goals in fitting
    class_weights = class_weight.compute_class_weight("balanced", classes=[0, 1], y=np.array(train_label))
    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(
        concat_input,
        train_label,
        steps_per_epoch=n_train // batch_size,
        epochs=max_epochs,
        validation_split=0.2,
        class_weight=class_weight_dict,
        batch_size=batch_size,
        callbacks=get_callbacks(patience=patience, monitor=monitor),
        verbose=2)
    return history


def vectorize_onehot(train_ds, test_ds, seq_len=1, verbose=True):
    # TextVectorization layers
    num_feats = num_nums(train_ds)
    train_lays, test_lays = [], []
    # Create tensors from raw input data that are numerical
    numputs = tf.constant([row[:num_feats] for row in train_ds])
    numputs_test = tf.constant([row[:num_feats] for row in test_ds])
    train_lays.append(tf.cast(numputs, tf.float32))
    test_lays.append(tf.cast(numputs_test, tf.float32))
    # Create text vectorization and one-hot encoded vectors
    for col_num in range(num_feats, len(train_ds[0])):
        num_types = len(set([row[col_num] for row in train_ds]))  # Number of unique outcomes for that variable
        # Vectorize layer for categorical variables
        text_vect = tf.keras.layers.TextVectorization(
            standardize="lower_and_strip_punctuation",
            max_tokens=num_types + 2,
            output_mode='int',  # To create unique integer indices for each token
            output_sequence_length=seq_len)
        # Adapt, apply and onehot encode the vectorization layer
        text_vect.adapt([row[col_num] for row in train_ds])
        if verbose:  # Show vectorization
            for i in range(1, num_types + 2):
                print(f"Vectorized into {i} means--> {text_vect.get_vocabulary()[i]}")
        one_hot_vocab = tf.one_hot(text_vect([row[col_num] for row in train_ds]), len(text_vect.get_vocabulary()))
        one_hot_test = tf.one_hot(text_vect([row[col_num] for row in test_ds]), len(text_vect.get_vocabulary()))
        one_hot_vocab = tf.squeeze(one_hot_vocab, axis=1)
        one_hot_test = tf.squeeze(one_hot_test, axis=1)
        train_lays.append(one_hot_vocab)
        test_lays.append(one_hot_test)

    # Create one big input tensor from one hot encodeds and numericals
    concat_input = tf.concat(train_lays, axis=-1)
    concat_test = tf.concat(test_lays, axis=-1)
    return concat_input, concat_test


def num_nums(training):
    nums = 0
    for entry in training[0]:
        if isinstance(entry, (int, float)):
            nums += 1
    return nums


'''
NEEDS REVISED
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


def create_models(numfeats, tiny=True, med=True, med_tanh=True):
    output = []
    if tiny:
        tiny_mod = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='elu', input_shape=(numfeats,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='elu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid")], name="Tiny")
        output.append(tiny_mod)
    if med:
        med_mod = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(numfeats,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid")], name="Medium")
        output.append(med_mod)
    if med_tanh:
        med_mod = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh', input_shape=(numfeats,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='tanh'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='tanh'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid")], name="Medium_tanh")
        output.append(med_mod)
    return output


def main():
    num_tests = 100  # Number of random shots to test
    size_histories = {}

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

    for inp, lab in zip([train_in_11, train_in_12, train_in_13, train_in_14, train_in_15, train_in_16, train_in_17,
                         train_in_18, train_in_19, train_in_20, train_in_21],
                        [train_lab_11, train_lab_12, train_lab_13, train_lab_14, train_lab_15, train_lab_16,
                         train_lab_17, train_lab_18, train_lab_19, train_lab_20, train_lab_21]):
        train_in.extend(inp)
        train_lab.extend(lab)

    test_in, test_lab = train_in_22, train_lab_22

    # Create Constant Tensors for ease of computing and leaving in/out tensors untouched
    onehotted_text = vectorize_onehot(train_ds=train_in, test_ds=test_in, verbose=False)
    concat_input, train_label = tf.constant(onehotted_text[0]), tf.constant(train_lab)
    concat_test, test_label = tf.constant(onehotted_text[1]), tf.constant(test_lab)

    numfeats = len(concat_input[0])
    rando = random.randint(10, len(test_lab) - num_tests)

    # Evaluate models with new unseen test data
    for mod in create_models(numfeats):
        # Compile and Train models and store progress
        fitted = compile_and_fit(mod, concat_input=concat_input, train_label=train_label)
        size_histories[f"{mod.name}"] = fitted
        # plot_loss(fitted, title=f"{mod.name} Training vs Validation Loss")
        loss, bin_entr, acc = mod.evaluate(concat_test, test_label, verbose=2)
        print(f"{mod.name} \n Evaluation loss, binary crossentropy, acc = {(loss, bin_entr, acc)}")

        # Prediction of some shots
        predictions = mod.predict(concat_test[rando:rando + num_tests])
        print("\n" + f"{mod.name} Predictions:")
        for pred, truth in zip(predictions, test_label[rando:rando + num_tests]):
            state = ('No Goal', 'Goal')[truth]
            print(f"Predicted: {pred[0] * 100:.2f}% vs Truth: {state}")

def main_test():
    num_tests = 100  # Number of random shots to test
    size_histories = {}

    train_in_11, train_lab_11 = get_shot_data(201102, 1230)
    train_in_12, train_lab_12 = get_shot_data(201202, 720)
    # train_in_13, train_lab_13 = get_shot_data(201302, 1230)
    # train_in_14, train_lab_14 = get_shot_data(201402, 1230)
    # train_in_15, train_lab_15 = get_shot_data(201502, 1230)
    # train_in_16, train_lab_16 = get_shot_data(201602, 1230)
    # train_in_17, train_lab_17 = get_shot_data(201702, 1271)
    # train_in_18, train_lab_18 = get_shot_data(201802, 1271)
    # train_in_19, train_lab_19 = get_shot_data(201902, 1082)
    # train_in_20, train_lab_20 = get_shot_data(202002, 868)
    # train_in_21, train_lab_21 = get_shot_data(202102, 1312)
    train_in_22, train_lab_22 = get_shot_data(202202, 1312)

    train_in, train_lab = list(train_in_11), list(train_lab_11)

    for inp, lab in zip([train_in_11, train_in_12],
                        [train_lab_11, train_lab_12]):
        train_in.extend(inp)
        train_lab.extend(lab)

    test_in, test_lab = train_in_22, train_lab_22

    # Create Constant Tensors for ease of computing and leaving in/out tensors untouched
    onehotted_text = vectorize_onehot(train_ds=train_in, test_ds=test_in, verbose=False)
    concat_input, train_label = tf.constant(onehotted_text[0]), tf.constant(train_lab)
    concat_test, test_label = tf.constant(onehotted_text[1]), tf.constant(test_lab)

    numfeats = len(concat_input[0])
    rando = random.randint(10, len(test_lab) - num_tests)

    # Evaluate models with new unseen test data
    for mod in create_models(numfeats):
        # Compile and Train models and store progress
        fitted = compile_and_fit(mod, concat_input=concat_input, train_label=train_label)
        size_histories[f"{mod.name}"] = fitted
        # plot_loss(fitted, title=f"{mod.name} Training vs Validation Loss")
        loss, bin_entr, acc = mod.evaluate(concat_test, test_label, verbose=2)
        print(f"{mod.name} \n Evaluation loss, binary crossentropy, acc = {(loss, bin_entr, acc)}")

        # Prediction of some shots
        predictions = mod.predict(concat_test[rando:rando + num_tests])
        print("\n" + f"{mod.name} Predictions:")
        for pred, truth in zip(predictions, test_label[rando:rando + num_tests]):
            state = ('No Goal', 'Goal')[truth]
            print(f"Predicted: {pred[0] * 100:.2f}% vs Truth: {state}")


if __name__ == "__main__":
    main()
