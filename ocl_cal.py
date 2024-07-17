import numpy as np
import torch

def oclusion (sentence, tag_list, word_idx_target, tag_target, trained_model):
    ocl_scores = []
    tam = len(sentence)
    tam_tag = len(trained_model.get_weights())
    for i in range(tam):
        ocl_scores_matrix = []
        copy_sent = sentence.copy()
        temp_sent = np.delete(copy_sent, i)
        temp_sent = np.append(temp_sent, [0])
        ocl_lstm_scores = trained_model.predict(np.array([temp_sent]))[0]
        for j in range(tam):
            if j < i:
                ocl_scores_matrix.append(ocl_lstm_scores[j].tolist())
            elif j > i:
                ocl_scores_matrix.append(ocl_lstm_scores[j-1].tolist())
            else:
                zeros = np.zeros(tam_tag).tolist()
                ocl_scores_matrix.append(zeros)
        ocl_scores.append(ocl_scores_matrix)
    ocl_scores = np.asarray(ocl_scores)

    p0 = trained_model.predict(np.array([sentence]))
    end_sent = np.where(sentence == 0)[0][0]
    pred_matrix = p0[0][:end_sent]

    ocl_matrix = []
    for i in range(end_sent):
        olc_sub_matrix = []
        for j in range(end_sent):
            olc_sub_matrix.append(ocl_scores[i][j])
        ocl_matrix.append(olc_sub_matrix)
    ocl_matrix = np.asarray(ocl_matrix)

    idxTag = tag_list.index(tag_target)

    delta_scores = pred_matrix-ocl_matrix[word_idx_target]
    scr_list = torch.from_numpy(delta_scores.T[idxTag])
    scr_list_abs = scr_list.abs()
    sorted_scr_list = torch.sort(scr_list_abs, descending =True)

    return sorted_scr_list