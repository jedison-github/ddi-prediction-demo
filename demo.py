from keras.models import model_from_json
import os
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
import random

from seq_self_attention import SeqSelfAttention
from config import Config
from load_data_ddi import load_test_data, sentence_split_for_pcnn

# ---------------------- Warning! ----------------------#
# 1. Following the requirements.txt might be required (Actually other version worked well)
# 2. seq_self_attention.py as custom object for model
# ------------------------------------------------------#


def f1_evaluate(loaded_model, test_data):
    te_sent_left, te_sent_mid, te_sent_right, te_d1_left, te_d1_mid, te_d1_right, te_d2_left, te_d2_mid, te_d2_right, te_y = test_data
    pred_te = loaded_model.predict(x=[te_sent_left, te_sent_mid, te_sent_right, te_d1_left, te_d1_mid, te_d1_right,
                                      te_d2_left, te_d2_mid, te_d2_right], batch_size=32, verbose=1)
    te_f1 = f1_score(np.argmax(te_y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
    te_p = precision_score(np.argmax(te_y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')
    te_r = recall_score(np.argmax(te_y, 1), np.argmax(pred_te, 1), [1, 2, 3, 4], average='micro')

    print('##test##, prec: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(te_p, te_r, te_f1))


if __name__ == "__main__":
    model_dir = 'model'

    # Load model from json file
    with open(os.path.join(model_dir, 'model.json'), 'r') as f:
        json_data = f.read()
    loaded_model = model_from_json(json_data, custom_objects={"SeqSelfAttention": SeqSelfAttention})

    # load weight from h5 file
    loaded_model.load_weights(os.path.join(model_dir, "weights.h5"))
    # print(loaded_model.summary())

    # load test data
    cfg = Config()
    (te_sentences2idx, te_d1_pos_lst, te_d2_pos_lst, te_pos_tuple_lst, te_y), (vocb, vocb_inv), (d1_vocb, d2_vocb), \
        (te_sentences, te_drug1_lst, te_drug2_lst, te_rel_lst) = load_test_data(unk_limit=cfg.unk_limit, max_sent_len=cfg.max_sent_len)

    (te_sent_left, te_d1_left, te_d2_left), (te_sent_mid, te_d1_mid, te_d2_mid), (te_sent_right, te_d1_right, te_d2_right) = \
        sentence_split_for_pcnn(sentences2idx=te_sentences2idx, d1_pos_lst=te_d1_pos_lst, d2_pos_lst=te_d1_pos_lst,
                                pos_tuple_lst=te_pos_tuple_lst, max_sent_len=cfg.max_sent_len)

    del te_sentences2idx, te_d1_pos_lst, te_d2_pos_lst, te_pos_tuple_lst

    # Run the model (Only with test data)
    f1_evaluate(loaded_model, (te_sent_left, te_sent_mid, te_sent_right, te_d1_left,
                               te_d1_mid, te_d1_right, te_d2_left, te_d2_mid, te_d2_right, te_y))

    #--------------------------------------------------------------------------------------------------------------------#
    # Predict random data from test set
    with open(os.path.join('data', 'test_orig_sent.txt'), 'r', encoding='utf-8') as f:
        orig_sent_lst = []
        for line in f:
            orig_sent_lst.append(line.rstrip())

    rel_class = ['false', 'advise', 'mechanism', 'effect', 'int']
    num_test_data = len(te_y)
    random_idx = random.choice(range(num_test_data))
    # Print info
    print("\n\n----------------------------------------------------------------")
    print("Random idx:", random_idx)
    print("Original Sentence:", orig_sent_lst[random_idx])
    # print("Drug replaced Sentence:", ' '.join(te_sentences[random_idx]))
    print("Drug 1:", te_drug1_lst[random_idx])
    print("Drug 2:", te_drug2_lst[random_idx])
    print()
    print("True Relation:", rel_class[np.argmax(te_y[random_idx]).item()])
    # prediction
    pred_te = loaded_model.predict(x=[te_sent_left[[random_idx], :], te_sent_mid[[random_idx], :], te_sent_right[[random_idx], :], te_d1_left[[random_idx], :], te_d1_mid[[random_idx], :], te_d1_right[[random_idx], :],
                                      te_d2_left[[random_idx], :], te_d2_mid[[random_idx], :], te_d2_right[[random_idx], :]], batch_size=1, verbose=0)
    print("Predicted Relation:", rel_class[int(np.argmax(pred_te))])
