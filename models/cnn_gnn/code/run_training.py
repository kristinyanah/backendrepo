import pickle
import random
import sys
import timeit
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_score, recall_score


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.sum(xs, 0), 0)

    def cnn(self, xs, i):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        hs = torch.relu(self.W_cnn[i](xs))
        return torch.squeeze(torch.squeeze(hs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        for i in range(layer):
            hs = self.cnn(xs, i)
            x = torch.relu(self.W_attention(x))
            hs = torch.relu(self.W_attention(hs))
            weights = torch.tanh(F.linear(x, hs))
            xs = torch.t(weights) * hs
        return torch.unsqueeze(torch.sum(xs, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        x_fingerprints = self.embed_fingerprint(fingerprints)
        x_compound = self.gnn(x_fingerprints, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        x_words = self.embed_word(words)
        x_protein = self.attention_cnn(x_compound, x_words, layer_cnn)

        # concatenates tensors in the given dimension (in this case 1)
        y_cat = torch.cat((x_compound, x_protein), 1)
        z_interaction = self.W_out(y_cat)

        return z_interaction

    def __call__(self, data, train=True):
        # ok so if just testing,
        # TODO: does interaction (-999) somehow fuck up result? shouldn't train. idk why we do forward algo here
        # t_interaction is the ground truth data i.e. true 1's (pos) and 0's (neg) of training data
        inputs, t_interaction = data[:-1], data[-1]
        # model outputs (presumably, the predictions) -- should read about exactly what those are in PyTorch
        z_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(z_interaction, t_interaction)
            return loss
        else:
            # gets the probabilities of the positive class (i.e. 1) occurring
            z = F.softmax(z_interaction, 1).to('cpu').data[0].numpy()
            t = int(t_interaction.to('cpu').data[0].numpy())
            return z, t


class Predictor(object):
    def __init__(self, model):
        """Load the pre-trained model from the directory of output/model."""
        self.model = model
        model.load_state_dict(torch.load('../output/model/' + setting))

    def predict(self, dataset, smiles_sequence_list):

        z_list, t_list = [], []
        for data in dataset:
            z = self.model(data)
            z_list.append(z[1])
            t_list.append(np.argmax(z))

        with open('prediction_result.txt', 'w') as f:
            f.write('smiles sequence '
                    'interaction_probability binary_class\n')
            for (c, p), z, t in zip(smiles_sequence_list, z_list, t_list):
                f.write(' '.join(map(str, [c, p, z, t])) + '\n')


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset, get_probabilities=False, iter=None, type=None):

        # if iter > 15:
        #     raise Exception("gotta end this mofo")

        z_list, t_list = [], []
        for data in dataset:
            # z looks like scores (predictions) for 0 and 1, and t is given interaction (i.e. "true" interaction, in testing case -999)
            z, t = self.model(data, train=False)
            z_list.append(z)
            t_list.append(t)

        # scores and labels are the probability outputs (likelihoods) and classifications (pos/neg)
        score_list, label_list = [], []
        for z in z_list:
            # prob of POSITIVE (i.e. class 1) score
            score_list.append(z[1])
            label_list.append(np.argmax(z))

        if get_probabilities:
            # returns the probability of a positive score, and the predicted label (based on which prob is higher)
            return(score_list, label_list)

        # scores here appear to be a probability estimate of the positive class!!!!!!!!!!!!!
        auc_score = roc_auc_score(t_list, score_list)
        # get true positive rate, false positive rate, and threshold
        fpr, tpr, thresholds = roc_curve(t_list, score_list)
        # probably remove, just wanna see if it's the same as above -- it is!!
        # new_auc = auc(fpr, tpr)

        # write info to file
        out = open("thresholds_full_BRENDA_0_to_point_1.txt", "a")
        for i, f in enumerate(fpr):
            # value based on looking at initial ROC curve
            if f > 0.10:
                print "Epoch is {}".format(iter)
                print "Type is: '{}'".format(type)
                print "fpr is {} at index {}, now bigger than 0.10".format(f, i)
                out.write("\nEpoch: {}\n".format(iter))
                out.write("\nType: {}\n\n".format(type))
                fpr_thresh = zip(fpr[0:i+60], thresholds[0:i+60])
                out.write("FPRs, Thresholds:\n")
                for pair in fpr_thresh:
                    out.write("{}\n".format(pair))
                out.write("\n\n")
                out.write("AUC:\n {:.3f}\n\n".format(auc_score))


                thresh = thresholds[i]
                print "thresh is {}".format(thresh)
                print "thresholds in that area are {}\n\n".format(thresholds[0:i+60])
                # stop looping once we get to the right FPR point
                break
        out.close()
        # plot that mf'ing curve -- currently plots all lines on same plot
        colors = ['g', 'b', 'r', 'c', 'm', 'k']
        # use mod to make diff color bins
        if iter:
            color = colors[iter % len(colors)]
        else:
            color = 'b'

        plt.title('Receiving Operator Characteristic (ROC)')
        _, ax = plt.subplots()
        plt.plot(fpr, tpr, color, label='AUC={:.3f}'.format(auc_score))
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.xticks(np.arange(0, 1, step=0.05))
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.025))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # plt.show()
        plt.savefig("full_BRENDA_0_to_point_1_ROC_new_epoch_{}_{}.png".format(iter, type or "no_type"))
        plt.close()

        precision = precision_score(t_list, label_list)
        recall = recall_score(t_list, label_list)

        return auc_score, precision, recall

    def result(self, epoch, time, loss, auc_dev,
               auc_test, precision, recall, file_name):
        with open(file_name, 'a') as f:
            result = map(str, [epoch, time, loss, auc_dev,
                               auc_test, precision, recall])
            f.write('\t'.join(result) + '\n')

    # save both the state dict, which is the recommended way of reloading a model, and the whole model, just in case
    def save_model(self, model, file_name):
        torch.save(model.state_dict(), file_name + ".pt")
        torch.save(model, file_name + "_whole_model.pt")


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def get_input_data(dir_input):
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)

    return (compounds, adjacencies, proteins, interactions)


def _split_dev_and_test_datasets(dataset, split, validation_ratio, test_ratio):
    """
    Splits the remainder of the data into the appropriate validation (dev) and test proportions.

    NOTE: to be run after the initial splitting off of the training set.

    :param dataset: The remainder of the dataset, after the training portion has been split off.
    :param split: The portion of the original dataset left to split up (i.e. if 0.80 was used for training, this would
        be 0.2)
    :param validation_ratio: The proportion of this dataset that should be the validation (dev) set.
    :param test_ratio: The proportion of this
    :return:
    """
    # get ratio of validation portion:test portion (e.g. 1:1)
    valid_to_test_ratio = (validation_ratio / test_ratio)
    # break up that test and validation set split into equal parts, i.e. valid_to_test_ratio + 1 parts (e.g. 2 parts)
    # and then get the portion of those chunks that should be test (e.g. .2 / 2 == .1)
    # (draw it out if it's confusing)
    portion_test = split / (valid_to_test_ratio + 1)
    # get portion that will be validation set, since that's what split_dataset() needs
    portion_valid = split - portion_test

    # return dev and test datasets, in that order
    return split_dataset(dataset, portion_valid)


def create_dataset(dir_input, training_ratio=0.80, validation_ratio=0.10, test_ratio=0.10):
    """
    Automatically split one read-in dataset (i.e. from input folder) into training, validation, and test sets, as designated.
    Also randomizes dataset, unless getting just a testset out of it. Default is (80% training, 10% validation, 10% test)

    User also has the option to pass in a testset-specific or trainingset-specific file.

    NOTE: ratios must add up to 1.0

    :param dir_input: Path to preprocessed input data folder; created in Preprocess_data.py
    :param training_ratio: Proportion of dataset (0.0-1.0) that should be used for training.
    :param validation_ratio: Proportion of dataset (0.0-1.0) that should be used for validation
    :param test_ratio: Proportion of dataset (0.0-1.0) that should be used for testing
    :return:
    """
    if (training_ratio + validation_ratio + test_ratio) != 1.0:
        raise Exception("Dataset splits must add up to 100%. Training ratio: {} Validation ratio: {} Test ratio: {}".
                        format(training_ratio, validation_ratio, test_ratio))

    # load preprocessed data
    compounds, adjacencies, proteins, interactions = get_input_data(dir_input)
    dataset = list(zip(compounds, adjacencies, proteins, interactions))

    # if we want it to be 100% testset, can just return it once it's loaded
    if training_ratio == 0.0 and validation_ratio == 0.0:
        return (None, None, dataset)

    # if we're not just testing, then shuffle
    # TODO: may wanna make this bool-controlled, in case we wanna shuffle training set but not test
    dataset = shuffle_dataset(dataset, 1234)

    # OG Tsubaki way, 80% train, 10% dev, 10% test
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)


    # TODO: error in this calculation, don't use it for now (got a 40:9:1 split?)
    # # otherwise, split data as appropriate
    # dataset_train, dataset_ = split_dataset(dataset, training_ratio)
    # # first get the portion of the set left to split (e.g. 20% if default)
    # test_and_valid_split = (1.0 - training_ratio)
    # # then split the set using appropriate proportions
    # dataset_dev, dataset_test = _split_dev_and_test_datasets(dataset_, test_and_valid_split, validation_ratio, test_ratio)

    return (dataset_train, dataset_dev, dataset_test)


def set_processor():
    """
    Sets processor to GPU if available, and CPU if not.
    :return: The appropriate PyTorch device type to be passed to the model.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    return device


if __name__ == "__main__":
    # read in variables from shell script
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, lr, lr_decay,
     decay_interval, iteration, saved_model, testset, setting) = sys.argv[1:]

    # if an already trained model is provided without a testset, it's unclear what to use for testing
    if saved_model and not testset:
        raise Exception ("Need to provide a test set via 'test_data' for saved model {}".format(saved_model))

    # put epoch timestamp at front so saved filenames include it
    setting = str(int(time.time())) + "_" + setting

    # convert strings to ints or floats as appropriate
    (dim, layer_gnn, window, layer_cnn, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, decay_interval,
                            iteration])
    lr, lr_decay = map(float, [lr, lr_decay])

    # set CPU or GPU usage
    device = set_processor()


    """
    
    Set up datasets used for training, validation (dev), and testing
    
    """

    # set input directory name based on DATASET, ngram, and radius
    # TODO: make this nicer, rearrange logic flow
    # if we've supplied a test set, use it for all of testing (regardless of whether or not we're training)
    if testset:
        test_dir_input = ('../dataset/' + DATASET + '/test_input/'
                 'radius' + radius + '_ngram' + ngram + '/' + testset + "/")

        # get the randomized test set
        _, _, dataset_test = create_dataset(test_dir_input, training_ratio=0.0, validation_ratio=0.0, test_ratio=1.0)

    # this will be training data and/or test data
    dir_input = ('../dataset/' + DATASET + '/input/'
                                           'radius' + radius + '_ngram' + ngram + '/')

    # if we haven't supplied a model i.e. if we're doing training
    if not saved_model:
        # if we're training but also supply a test set, i.e. want all of OG data set used for training and validation
        if testset:
            print "Testset supplied, {} will only be used for training and validation".format(dir_input)
            # get the randomized data sets for just training and dev
            dataset_train, dataset_dev, _ = create_dataset(dir_input, training_ratio=0.90,
                                                                      validation_ratio=0.10, test_ratio=0.0)
        # if we're training and did not supply a test set
        else:
            print "No testset supplied, will use {} dataset for training, validation, and dev".format(dir_input)
            # get the randomized data sets w default proportions
            dataset_train, dataset_dev, dataset_test = create_dataset(dir_input)


    # TODO ensure this is ok!!! need to know if n_fingerprint and n_word are used for anything other than training
    # actually, looks like n_word affecte embed_word() which affects forward() output, which is indeed called in testing? don't remember why I thought this was fine
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict) + 1
    n_word = len(word_dict) + 1

    """
    
    Set up model itself, as well as Tester and Trainer class (if necessary). Also set up output files
    
    """
    torch.manual_seed(1234)
    model = CompoundProteinInteractionPrediction().to(device)
    # NOTE: if loading something trained on GPU, need to load it special on CPU
    if saved_model:
        print "Loading saved model {}; no training will take place, only testing".format(saved_model)
        # TODO: remove --specific to saving on GPU and loading on CPU

        model.load_state_dict(torch.load(saved_model, map_location=device))
        # not going to do training, so setup for test (sets 'training' attribute to false)
        model.eval()
    else:
        print "Will perform training"
        trainer = Trainer(model)

    tester = Tester(model)

    # set filepath for the model to be saved
    file_model = '../output/model/' + setting

    # print('Training...')
    print('Epoch Time(sec) Loss_train AUC_dev '
          'AUC_test Precision_test Recall_test')

    start = timeit.default_timer()

    file_result = '../output/result/' + setting + '.txt'


    """
    
    Train and/or test
    
    """
    # if we're training
    if not saved_model:
        # start to write output result file
        print "Initializing training output file"
        with open(file_result, 'w') as f:
            f.write('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
                    'AUC_test\tPrecision_test\tRecall_test\n')
        print "Beginning training, w/intermittent testing"

        for epoch in range(iteration):
            if (epoch+1) % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay

            loss = trainer.train(dataset_train)


            # TODO: uncomment this!!!!!!
            # auc_dev = None
            auc_dev = tester.test(dataset_dev, iter=epoch, type="dev")[0]
            auc_test, precision, recall = tester.test(dataset_test, iter=epoch, type="test")

            end = timeit.default_timer()
            time = end - start

            tester.result(epoch, time, loss, auc_dev,
                          auc_test, precision, recall, file_result)
            tester.save_model(model, file_model)

            print(epoch, time, loss, auc_dev, auc_test, precision, recall)
    # if we're just testing
    else:
        # scores == positive probabilities, labels == 0 | 1
        scores, labels = tester.test(dataset_test, get_probabilities=True)
#########
        ## if I want to write that overview file of the testing stats
        # auc_test, precision, recall = tester.test(dataset_test)
        #
        # end = timeit.default_timer()
        # time = end - start
        #
        # tester.result("--", time, "--", "--",
        #               auc_test, precision, recall, file_result)

        ########

        # ah HERE write to file! yes.
        with open(file_result, "w") as output:
            # loop through test input file, outputting appropriate input line with each score/label pair
            with open("../dataset/" + DATASET + "/test_original/" + testset + ".txt", "r") as test_input:
                for i, line in enumerate(test_input):
                    # if the likelihood is 75% or less (as determined from ROC curve), then it's actually more likely to be a negative interaction (0)
                    if scores[i] <= 0.75:
                        # wanna set label ourselves, don't wanna use their labels at all
                        # score = 1.0 - scores[i]
                        interaction = 0
                    # otherwise it's a positive interaction
                    else:
                        # score = scores[i]
                        interaction = 1
                    # add line of input file, along with probability and predicted interaction based on threshold
                    out_line = line.rstrip() + " " + str(scores[i]) + " " + str(interaction) + "\n"
                    output.write(out_line)



