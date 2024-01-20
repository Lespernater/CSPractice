import math
import requests
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import *
from sklearn.utils import class_weight
import pandas as pd


class Event:
    """
    Play object using play-by-play from NHL API
    """
    def __init__(self, dict_from_api, prev_event=None):
        self.whole = dict_from_api
        self.clock = self.sec_remain()
        self.typev = self.whole.get('typeDescKey', "NULL")
        self.period_dic = self.whole.get('periodDescriptor', None)
        # self.home_sided = self.whole['homeTeamDefendingSide']
        self.details = self.get_details()
        self.team_owner = self.get_teamowner()
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

    def get_period(self):
        if self.period_dic:
            return str(self.period_dic['number'])
        return str(0)

    def locate(self):
        """
        Calculate angle and distance from net using x-coord and y-coord on ice and return usable vector including zone

        :return: list of x-coord (int), y-coord (int), distance (int), angle (int), zone (str)
        """
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
        """
        Output stats from event object into vector where ints are before strings

        :return: Usable vector of stats for event object (list)
        """
        if self.prev_event:
            outputs = combine_elements(self.loc_vect, self.prev_event.get_loc(), self.speed(),
                                       self.prev_event.time_since_prev, self.clock, self.prev_event.typev,
                                       self.play1, self.play2, self.get_period(), self.shot_method, self.team_owner,
                                       self.situation)
        else:
            outputs = combine_elements(self.loc_vect, [-1, -1, -1, -1, "NA"], self.speed(), -1, self.clock,
                                       "NULL", self.play1, self.play2, self.get_period(), self.shot_method,
                                       self.team_owner, self.situation)
        return outputs

    def sec_remain(self):
        """
        Preprocess time into period_dic into number of seconds into period_dic as int

        :return: number seconds into the period_dic as int
        """
        mins, sec = str(self.whole['timeRemaining']).split(":")
        seconds = int(mins) * 60 + int(sec)
        return seconds

    def speed(self):
        """
        Calculate angular and linear speed between current event and previous event

        :return: angular and linear_speed
        """
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
            return self.prev_event.get_clock() - self.clock
        return -1

    def get_details(self):
        if 'details' in self.whole:
            return self.whole['details']
        return None

    def get_teamowner(self):
        if self.details:
            return str(self.details.get('eventOwnerTeamId', "NULL"))
        return "NULL"

    def update_players(self):
        """
        Fetch players (player 1 and player 2) involved in event. If not found, use 0 in its place

        :return: Tuple of player1 and player2 (strings)
        """
        out = [0, 0]
        if self.details:
            if self.typev in ('takeaway', 'giveaway'):
                out = self.details.get('playerId', 0), 0
            elif self.typev == 'hit':
                out = self.details.get('hittingPlayerId', 0), self.details.get('hitteePlayerId', 0)
            elif self.typev == 'blocked-shot':
                out = self.details.get('blockingPlayerId', 0), self.details.get('shootingPlayerId', 0)
            elif self.typev == 'faceoff':
                out = self.details.get('winningPlayerId', 0), self.details.get('losingPlayerId', 0)
            elif self.typev in ('missed-shot', 'shot-on-goal'):
                out = self.details.get('shootingPlayerId', 0), self.details.get('goalieInNetId', 0)
            elif self.typev == 'goal':
                out = self.details.get('scoringPlayerId', 0), self.details.get('goalieInNetId', 0)
        return str(out[0]), str(out[1])

    def update_shot(self):
        return self.typev in ("blocked-shot", "missed-shot", "goal", "miss", "shot-on-goal")


def combine_elements(*args):
    """
    Reorder vector so ints are before strings

    :param args: any number of arguments in a list
    :return: list where order is remained except ints are before others
    """
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


def get_shot_data_current(start=1):
    """
    Process all shot data throughout reg season from current year starting from start

    :param start: Game number in season to start processing shot data
    :return: tuple of raw input shot data and raw labels for shot data
    """
    nn_in = []
    nn_out = []
    input_year_02 = str(int(datetime.now().year) - 1) + "02"
    # Get JSON game data from NHL api for (default=every) game(s) in that season
    unplayed = False
    while not unplayed:
        game_id = str(input_year_02) + f"{start:04}"
        print(f"Getting stats from game {game_id}")
        game_url = "https://api-web.nhle.com/v1/gamecenter/" + game_id + "/play-by-play"
        response = requests.get(game_url, params={"Content-Type": "application/json"})
        data = response.json()
        response.close()
        year, month, day = data["gameDate"].split("-")
        if datetime.now() < (datetime(int(year), int(month), int(day)) + timedelta(days=1)):
            unplayed = True
        start += 1
        if 'plays' in data:
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
    :return: (None) - plt.show()
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
                    max_epochs=1000, patience=8, monitor='val_binary_crossentropy', plot=True):
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

    if plot:  # Plot learning rate change over epochs
        plot_lr(lr_sched)

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
    """
    Determine number of features that are ints

    :param training: input vector (from get_shot_data())
    :return: number of features that are ints (int)
    """
    nums = 0
    for entry in training[0]:
        if isinstance(entry, (int, float)):
            nums += 1
    return nums


def create_models(numfeats, tiny=True, med=True, med_tanh=True, large_sig=True):
    """

    :param numfeats: number of features in
    :param tiny:
    :param med:
    :param med_tanh:
    :param large_tanh:
    :return:
    """
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
    if large_sig:
        med_mod = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(numfeats,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='sigmoid'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2048, activation='sigmoid'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='sigmoid'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid")], name="Large_sigmoid")
        output.append(med_mod)
    return output


def data_parse_to_csv():
    train_in_11, train_lab_11 = get_shot_data(input_year_02=201102, num_games=1230)
    train_in_12, train_lab_12 = get_shot_data(input_year_02=201202, num_games=720)
    train_in_13, train_lab_13 = get_shot_data(input_year_02=201302, num_games=1230)
    train_in_14, train_lab_14 = get_shot_data(input_year_02=201402, num_games=1230)
    train_in_15, train_lab_15 = get_shot_data(input_year_02=201502, num_games=1230)
    train_in_16, train_lab_16 = get_shot_data(input_year_02=201602, num_games=1230)
    train_in_17, train_lab_17 = get_shot_data(input_year_02=201702, num_games=1271)
    train_in_18, train_lab_18 = get_shot_data(input_year_02=201802, num_games=1271)
    train_in_19, train_lab_19 = get_shot_data(input_year_02=201902, num_games=1082)
    train_in_20, train_lab_20 = get_shot_data(input_year_02=202002, num_games=868)
    train_in_21, train_lab_21 = get_shot_data(input_year_02=202102, num_games=1312)
    train_in_22, train_lab_22 = get_shot_data(input_year_02=202202, num_games=1312)
    train_in_23, train_lab_23 = get_shot_data_current(start=1)

    train_in, train_lab = list(train_in_11), list(train_lab_11)

    for inp, lab in zip([train_in_11, train_in_12, train_in_13, train_in_14, train_in_15, train_in_16, train_in_17,
                         train_in_18, train_in_19, train_in_20, train_in_21, train_in_22],
                        [train_lab_11, train_lab_12, train_lab_13, train_lab_14, train_lab_15, train_lab_16,
                         train_lab_17, train_lab_18, train_lab_19, train_lab_20, train_lab_21, train_lab_22]):
        train_in.extend(inp)
        train_lab.extend(lab)

    test_in, test_lab = list(train_in_23), list(train_lab_23)

    columns = ["Event x-coord", "Event y-coord", "Event distance", "Event angle", "Prev x-coord", "Prev y-coord",
               "Prev distance", "Prev angle", "Event anglular speed", "Event linear speed", "Time since prev event",
               "Seconds remaining period", "Event zone", "Prev zone", "Prev type", "Player1", "Player2",
               "Period", "Shot method", "Event team owner", "One-ice situation"]

    training_df = pd.DataFrame(train_in, columns=columns)
    testing_df = pd.DataFrame(test_in, columns=columns)
    training_lab_df = pd.DataFrame(train_lab, columns=["Goal"])
    testing_lab_df = pd.DataFrame(test_lab, columns=["Goal"])
    training_df.to_csv("training11-22.csv", index=True)
    testing_df.to_csv("testing23.csv", index=True)
    training_lab_df.to_csv("labs_training11-22.csv", index=True)
    testing_lab_df.to_csv("labs_testing23.csv", index=True)


def recreate_dataset(csv_file, full, num_intfeats=12, out_len=1000):
    with open(csv_file, "r") as file:
        output = []
        for line in file.readlines()[1:]:
            row = line.split(",")[1:]
            if len(row) > 1 :
                row = [int(item) for item in row[:num_intfeats]] + [str(item) for item in row[num_intfeats:]]
            else:
                row = int(row[0])
            output.append(row)
    if full:
        return output
    return output[0:out_len]


def update_current_season(csv_file):
    train_in_23, train_lab_23 = get_shot_data_current(start=1)
    columns = ["Event x-coord", "Event y-coord", "Event distance", "Event angle", "Prev x-coord", "Prev y-coord",
               "Prev distance", "Prev angle", "Event anglular speed", "Event linear speed", "Time since prev event",
               "Seconds remaining period", "Event zone", "Prev zone", "Prev type", "Player1", "Player2",
               "Period", "Shot method", "Event team owner", "One-ice situation"]
    testing_in_df = pd.DataFrame(train_in_23, columns=columns)
    testing_lab_df = pd.DataFrame(train_lab_23, columns=["Goal"])
    testing_in_df.to_csv(csv_file, index=True)
    testing_lab_df.to_csv(f"labs_{csv_file}", index=True)


def build_train_eval(train, train_lab, test, test_lab, num_tests=100, show_predictions=True, plot=True):
    size_histories = {}
    rando = random.randint(10, len(test_lab) - num_tests)
    print(f"Number of features: {len(train[0])}")

    # Evaluate models with new unseen test data
    for mod in create_models(len(train[0]), tiny=True, med=True, med_tanh=True, large_sig=True):
        # Compile and Train models and store progress
        fitted = compile_and_fit(mod, concat_input=train, train_label=train_lab, patience=20, plot=False)
        size_histories[f"{mod.name}"] = fitted
        if plot:
            plot_loss(fitted, title=f"{mod.name} Training vs Validation Loss")
        loss, bin_entr, acc = mod.evaluate(test, test_lab, verbose=2)
        print(f"{mod.name} \n Evaluation loss, binary crossentropy, acc = {(loss, bin_entr, acc)}")
        if show_predictions:
            # Prediction of some shots
            predictions = mod.predict(test[rando:rando + num_tests])
            print("\n" + f"{mod.name} Predictions:")
            for pred, truth in zip(predictions, test_lab[rando:rando + num_tests]):
                state = ('No Goal', 'Goal')[truth]
                print(f"Predicted: {pred[0] * 100:.2f}% vs Truth: {state}")


def main():
    num_tests = 100  # Number of random shots to test
    size_histories = {}

    train_in_11, train_lab_11 = get_shot_data(input_year_02=201102, num_games=1230)
    train_in_12, train_lab_12 = get_shot_data(input_year_02=201202, num_games=720)
    train_in_13, train_lab_13 = get_shot_data(input_year_02=201302, num_games=1230)
    train_in_14, train_lab_14 = get_shot_data(input_year_02=201402, num_games=1230)
    train_in_15, train_lab_15 = get_shot_data(input_year_02=201502, num_games=1230)
    train_in_16, train_lab_16 = get_shot_data(input_year_02=201602, num_games=1230)
    train_in_17, train_lab_17 = get_shot_data(input_year_02=201702, num_games=1271)
    train_in_18, train_lab_18 = get_shot_data(input_year_02=201802, num_games=1271)
    train_in_19, train_lab_19 = get_shot_data(input_year_02=201902, num_games=1082)
    train_in_20, train_lab_20 = get_shot_data(input_year_02=202002, num_games=868)
    train_in_21, train_lab_21 = get_shot_data(input_year_02=202102, num_games=1312)
    train_in_22, train_lab_22 = get_shot_data(input_year_02=202202, num_games=1312)
    train_in_23, train_lab_23 = get_shot_data_current(start=1)

    train_in, train_lab = list(train_in_11), list(train_lab_11)

    for inp, lab in zip([train_in_11, train_in_12, train_in_13, train_in_14, train_in_15, train_in_16, train_in_17,
                         train_in_18, train_in_19, train_in_20, train_in_21, train_in_22],
                        [train_lab_11, train_lab_12, train_lab_13, train_lab_14, train_lab_15, train_lab_16,
                         train_lab_17, train_lab_18, train_lab_19, train_lab_20, train_lab_21,train_lab_22]):
        train_in.extend(inp)
        train_lab.extend(lab)

    test_in, test_lab = train_in_23, train_lab_23

    # Create Constant Tensors for ease of computing and leaving in/out tensors untouched
    onehotted_input, onehotted_test = vectorize_onehot(train_ds=train_in, test_ds=test_in, verbose=False)
    concat_input, train_label = tf.constant(onehotted_input), tf.constant(train_lab)
    concat_test, test_label = tf.constant(onehotted_test), tf.constant(test_lab)

    numfeats = len(concat_input[0])
    print(f"Number of features: {numfeats}")
    rando = random.randint(10, len(test_lab) - num_tests)

    # Evaluate models with new unseen test data
    for mod in create_models(numfeats, tiny=True, med=False, med_tanh=False, large_sig=True):
        # Compile and Train models and store progress
        fitted = compile_and_fit(mod, concat_input=concat_input, train_label=train_label, patience=20)
        size_histories[f"{mod.name}"] = fitted
        plot_loss(fitted, title=f"{mod.name} Training vs Validation Loss")
        loss, bin_entr, acc = mod.evaluate(concat_test, test_label, verbose=2)
        print(f"{mod.name} \n Evaluation loss, binary crossentropy, acc = {(loss, bin_entr, acc)}")

        # Prediction of some shots
        predictions = mod.predict(concat_test[rando:rando + num_tests])
        print("\n" + f"{mod.name} Predictions:")
        for pred, truth in zip(predictions, test_label[rando:rando + num_tests]):
            state = ('No Goal', 'Goal')[truth]
            print(f"Predicted: {pred[0] * 100:.2f}% vs Truth: {state}")


def main_test():
    # Import NHL shot data from previous run
    train_in = recreate_dataset("training11-22.csv", full=True)
    test_in = recreate_dataset("testing23.csv", full=True)
    train_lab = recreate_dataset("training_lab_11-22.csv", full=True)
    test_lab = recreate_dataset("testing_lab_23.csv", full=True)
    # Text vectorize and one-hot
    onehotted_input, onehotted_test = vectorize_onehot(train_ds=train_in, test_ds=test_in, verbose=False)
    # Create Constant Tensors for ease of computing and leaving in/out tensors untouched
    concat_input, train_label = tf.constant(onehotted_input), tf.constant(train_lab)
    concat_test, test_label = tf.constant(onehotted_test), tf.constant(test_lab)
    # Modelling
    build_train_eval(concat_input, train_label, concat_test, test_label, show_predictions=True, plot=False)


if __name__ == "__main__":
    main_test()


