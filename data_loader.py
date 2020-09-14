import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data():
    speech_train = np.load('train.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('dev.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('test.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('dev_transcripts.npy', allow_pickle=True,encoding='bytes')

    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid


'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter2index):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter2index: Dictionary containing letter keys and their corresponding indices as values
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    letter_to_index_list = []

    for i in range(transcript.shape[0]):
        for j in range(len(transcript[i])):
            index = letter2index[chr(transcript[i][j])]
            letter_to_index_list.append(index)
        index = letter2index[' ']
        letter_to_index_list.append(index)

    return letter_to_index_list


def create_vocab_index(letter_list):
    letter2index = dict()

    for i in range(len(letter_list)):
        letter2index[letter_list[i]] = i

    return letter2index

class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours.
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train(batch_data):
    ### Return the packed padded speech and text data, and the length of utterance and transcript ###
    speech = []
    text = []

    for i in range(len(batch_data)):
        s , t = batch_data[i]
        speech.append(s)
        text.append(t.long())

    speech_lens = torch.LongTensor([len(seq) for seq in speech])
    text_lens = torch.LongTensor([len(seq) for seq in text])

    speech = torch.nn.utils.rnn.pad_sequence(speech, batch_first = True)
    text = torch.nn.utils.rnn.pad_sequence(text, batch_first = True)

    return speech,speech_lens,text,text_lens


def collate_test(batch_data):
    ### Return padded speech and length of utterance ###
    speech = []
    text = []

    for i in range(len(batch_data)):
        s, t = batch_data[i]
        speech.append(s)
        text.append(t.long())

    speech_lens = torch.LongTensor([len(seq) for seq in speech])

    speech = torch.nn.utils.rnn.pad_sequence(speech, batch_first=True)
    text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)

    return speech, speech_lens, text