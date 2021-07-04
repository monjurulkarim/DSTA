import numpy as np
import shutil
import glob
import os
b = np.load('output/UString/vgg16/dad/test_boostin_balanced_dataa_19_feb_epoch19/pred_res.npz', allow_pickle=True)
# b = np.load('output/UString/vgg16/dad/test_ist_iteration/pred_res.npz')
c= (b['vis_data'])

prediction = []
labels = []
toas = []
video_ids = []
for i in c:
#     inner_pred = []
    pred_item= i['pred_frames']
    label = i['label']
    video_id = i['video_ids']
    toa = i['toa']
    prediction.append(pred_item)
    labels.append(label)
    video_ids.append(video_id)
    toas.append(toa)

video_score=[]
for i in range(len(prediction)):
    for j in range(10):
        vid_score = max(prediction[i][j][:int(toas[i][j])])
        video_score.append(vid_score)

new_labels = []
for items in labels:
    items= np.array(items).tolist()
    for i in items:
        new_labels.append(i)

new_videos = []
for items in video_ids:
    items= np.array(items).tolist()
    for i in items:
        new_videos.append(i)

#getting the labels of the files from the directory
source_dir = 'data/dad/vgg16_features/testing/*'
vid_id = []
labels = []
for file in glob.glob(source_dir):
    b = np.load(file, allow_pickle=True)
    vid_id.append(b['ID'])
    labels.append(b['labels'])
print('#############################')
print('All the files have been loaded.')
Threshold = []
Recall = []
Precision = []
print('###############################')
print('Calculating recall for Different threshold: ')
for Th in np.arange(0, 1.0, 0.001):
    video_label_pred = [1 if video_score_ >= Th else 0 for video_score_ in video_score]
    #incorrect prediction
    incorrect = np.where(np.not_equal(video_label_pred, new_labels))
    #correct prediction
    correct = np.where(np.equal(video_label_pred, new_labels))
    #pulling the incorrecly predicted video ids
    incorrect_vid_id= []
    for i in incorrect[0]:
        incorrect_vid_id.append(new_videos[i])

    #pulling the incorrecly predicted video ids
    correct_vid_id= []
    for i in correct[0]:
        correct_vid_id.append(new_videos[i])

    F_positive=[]
    F_negative=[]
    for i in incorrect_vid_id:
        if i in vid_id:
            c= vid_id.index(i)
            label = labels[c]
            if label[1]> 0:
                F_negative.append(i)
            else:
                F_positive.append(i)

    T_positive=[]
    T_negative=[]
    for i in correct_vid_id:
        if i in vid_id:
            c= vid_id.index(i)
            label = labels[c]
            if label[1]> 0:
                T_positive.append(i)
            else:
                T_negative.append(i)

    pre = len(T_positive)/(len(T_positive)+ len(F_positive))
    rec = len(T_positive)/(len(T_positive)+ len(F_negative))
    Threshold.append(Th)
    Precision.append(pre)
    Recall.append(rec)

Recall = np.array(Recall)
a = np.where(Recall<=0.8)
print('index = ', a[0][0])
T = Threshold[a[0][0]]
# print(a)
print('recall :' , Recall)
print('====================')
print("Threshold value at 80% Recall : ", T)
