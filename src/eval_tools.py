import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline

#calculation of mTTA during training epoch
def evaluation_train(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    # Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    # new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         # new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("mean Time to accident at this training epoch= %.4f"%(mTTA))

    #not necessary
    # sort_time = new_Time[np.argsort(new_Recall)]
    # sort_recall = np.sort(new_Recall)
    # TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    # print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return mTTA


def evaluation(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    a = np.where(new_Recall>=0.8)
    P_R80 = new_Precision[a[0][0]]
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    print("Precision at Recall 80: %.4f"%(P_R80))
    print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return AP, mTTA, TTA_R80, P_R80

#please delete "_P_R80"
def evaluation_P_R80(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    a = np.where(new_Recall>=0.8)
    P_R80 = new_Precision[a[0][0]]
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    print("Precision at Recall 80: %.4f"%(P_R80))
    print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return AP, mTTA, TTA_R80, P_R80


def print_results(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, result_dir):
    result_file = os.path.join(result_dir, 'eval_all.txt')
    with open(result_file, 'w') as f:
        for e, APvid, AP, mTTA, TTA_R80 in zip(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all):
            f.writelines('Epoch: %s,'%(e) + ' APvid={:.3f}, AP={:.3f}, mTTA={:.3f}, TTA_R80={:.3f}\n'.format(APvid, AP, mTTA, TTA_R80))
    f.close()


def vis_results(vis_data, batch_size, vis_dir, smooth=False, vis_batchnum=2):
    pass
