from data_loader import *
from train_test import *
from LAS import *
import pandas as pd

def main():
    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()

    print("Loaded Data")

    LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
                   'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

    letter2index = create_vocab_index(LETTER_LIST)

    for i in range(transcript_train.shape[0]):
        transcript_train[i] = np.array(transform_letter_to_index(transcript_train[i], letter2index))

    for i in range(transcript_valid.shape[0]):
        transcript_valid[i] = np.array(transform_letter_to_index(transcript_valid[i], letter2index))

    transcript_test = np.zeros((speech_test.shape[0],1))
    transcript_test[:] = len(LETTER_LIST)-2

    train_dataset = Speech2TextDataset(speech_train,transcript_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_train)

    val_dataset = Speech2TextDataset(speech_valid,transcript_valid)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_train)

    test_dataset = Speech2TextDataset(speech_test,transcript_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_test)

    model = LAS(len(LETTER_LIST), 40, 256, 256, 512, 128, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=4, verbose=True)
    criterion = nn.CrossEntropyLoss(reduction="none")

    tf_prob = 0.05

    print("Started Training")

    for epoch in range(50):
        print("tf_prob:", tf_prob)
        # attentions,text_lens = train_data(model, train_loader,criterion,optimizer,tf_prob)
        loss = train(model, train_loader, criterion, optimizer, tf_prob)
        print("Loss:", loss)
        scheduler.step(loss)
        if ((epoch + 1) % 2 == 0) and (tf_prob < 0.50):
            tf_prob += 0.05
        torch.save(model, "dec_model.pt")

    submission = pd.read_csv("test_sample_submission.csv")
    test(model, test_loader, LETTER_LIST, submission)
    submission.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    main()