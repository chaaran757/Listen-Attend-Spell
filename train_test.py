from tqdm import tqdm
import numpy as np
import torch

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def train(model, train_loader, criterion, optimizer, tf_prob):
    model.train()
    model.to(device)
    # 1) Iterate through your loader
    progress_bar = enumerate(tqdm(train_loader))
    epoch_loss = 0
    for i,(speech,speech_lens,text,text_lens) in progress_bar:
# 3) Set the inputs to the device.
        speech = speech.to(device)
        speech_lens = speech_lens.to(device)
        text = text.to(device)
        text_lens = text_lens.to(device)
# 4) Pass your inputs, and length of speech into the model.
        loss= model(speech,speech_lens,text.float(),criterion,text_lens,tf_prob = tf_prob)
        loss.backward()
        #p = plot_grad_flow(model.named_parameters())
        masked_loss = loss/((text_lens+1).sum())
        print(masked_loss)
# 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
# 11) Take a step with your optimizer
        optimizer.step()
        epoch_loss += masked_loss
    epoch_loss = epoch_loss/len(train_loader)
    return epoch_loss


def test(model, test_loader, LETTER_LIST, submission):
    ### Write your test code here! ###
    model.eval()
    model.to(device)
    # 1) Iterate through your loader
    indx = 0
    progress_bar = enumerate(tqdm(test_loader))
    for i,(speech,speech_lens,text) in progress_bar:
        # 3) Set the inputs to the device.
        speech = speech.to(device)
        speech_lens = speech_lens.to(device)
        text = text.to(device)
        # 4) Pass your inputs, and length of speech into the model.
        char_indices = model(speech, speech_lens,text, is_train = False)

        char_indices = torch.stack(char_indices, dim=1)

        for j in range(char_indices.shape[0]):
            transcript = []
            for i in range(char_indices[j].shape[0]):
                if char_indices[j][i] != len(LETTER_LIST) - 1:
                    transcript.append(char_indices[j][i].cpu().numpy())

            for i in range(len(transcript)):
                transcript[i] = LETTER_LIST[transcript[i]]

            submission["Predicted"][indx] = ''.join(transcript[i] for i in range(len(transcript)))
            indx += 1