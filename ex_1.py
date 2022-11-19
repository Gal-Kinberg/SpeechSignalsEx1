import numpy as np
import librosa
from os import listdir


class SpeechTrainExample:
    def __init__(self, sample, label):
        self.sample = sample
        self.label = label


class SpeechClassifier1NN:
    def __init__(self, train_set_path):
        # load training examples
        self.examples = []

        for number_str, number in zip(['one', 'two', 'three', 'four', 'five'], [1, 2, 3, 4, 5]):
            dir_path = f'{train_set_path}/{number_str}/'
            for file_name in listdir(dir_path):
                if file_name == '.DS_Store':
                    continue
                y, mfcc = SpeechClassifier1NN.load_sample(f'{dir_path}/{file_name}')
                self.examples.append(SpeechTrainExample(sample=mfcc, label=number))

        self.nExamples = len(self.examples)
        return

    @staticmethod
    def DTW_distance(x, y):
        Nx = len(x)
        Ny = len(y)

        # initialize the DTW matrix to infinity
        DTW = np.inf * np.ones(shape=(Nx + 1, Ny + 1))
        DTW[0, 0] = 0

        # compute the DTW recursively
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                DTW[i, j] = SpeechClassifier1NN.euclidean_distance(x[i - 1], y[j - 1]) + np.min(
                    [DTW[i, j - 1], DTW[i - 1, j], DTW[i - 1, j - 1]])

        return DTW[Nx, Ny]

    @staticmethod
    def euclidean_distance(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    @staticmethod
    def load_sample(sample_path: str):
        y, sr = librosa.load(sample_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        return y, mfcc

    def classify_sample(self, sample_path):
        sample, mfcc = SpeechClassifier1NN.load_sample(sample_path)

        DTW_best_score = np.inf
        DTW_label = ''
        euclidean_best_score = np.inf
        euclidean_label = ''
        for iExample in range(self.nExamples):
            DTW_score = SpeechClassifier1NN.DTW_distance(mfcc, self.examples[iExample].sample)
            euclidean_score = SpeechClassifier1NN.euclidean_distance(mfcc, self.examples[iExample].sample)

            if DTW_score < DTW_best_score:
                DTW_best_score = DTW_score
                DTW_label = self.examples[iExample].label

            if euclidean_score < euclidean_best_score:
                euclidean_best_score = euclidean_score
                euclidean_label = self.examples[iExample].label

        return euclidean_label, DTW_label


if __name__ == '__main__':
    classifier = SpeechClassifier1NN(train_set_path='train_data')
    for iSample in range(1, 254):
        euclidean_label, DTW_label = classifier.classify_sample(f'test_files/sample{iSample}.wav')
        print(f'sample{iSample}.wav - {euclidean_label} - {DTW_label}')
