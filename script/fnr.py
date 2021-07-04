import numpy as np
import shutil
import glob
import os
b = np.load('output/UString/vgg16/dad/test_boostin_balanced_dataa_19_feb/pred_res.npz', allow_pickle=True)
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

threshold =0.583
video_label_pred = [1 if video_score_ >= threshold else 0 for video_score_ in video_score]
print('Threshold : ', threshold)

#incorrect prediction
incorrect = np.where(np.not_equal(video_label_pred, new_labels))
print('Total incorrect prediction : ', len(incorrect[0]))
print('++++++++++++++++++++++++++++++++')

#incorrect prediction
correct = np.where(np.equal(video_label_pred, new_labels))
print('Total correct prediction : ', len(correct[0]))
print('++++++++++++++++++++++++++++++++')


#pulling the incorrecly predicted video ids
incorrect_vid_id= []
for i in incorrect[0]:
    incorrect_vid_id.append(new_videos[i])

# print(incorrect_vid_id)

#pulling the incorrecly predicted video ids
correct_vid_id= []
for i in correct[0]:
    correct_vid_id.append(new_videos[i])

train_directory = 'data/boosting/dad/vgg16_features/testing/'
destination = 'data/boosting/dad/vgg16_features/from_2nd_ita_training/'
FN = os.path.join(destination,'FN')
FP = os.path.join(destination,'FP')
TN = os.path.join(destination,'TN')
TP = os.path.join(destination,'TP')

F_positive=[]
F_negative=[]
for i in incorrect_vid_id:
    d1 = str(i[0:4])
    d2 = '_1'
    d3 = str(i[5:11])
    d = d1 + d2 + d3
    # print(d)
    # file_name = train_directory +d + '.npz'
    file_name = train_directory +i + '.npz'
    b = np.load(file_name, allow_pickle=True)
    label = b['labels']
    if label[1]> 0:
        dest = FN +'/' + d1+ '_1'+ d3 + '.npz'
        # dest = FN +'/' + i + '.npz'
        F_negative.append(i)
        shutil.copyfile(file_name, dest)
    else:
        # dest = FP+'/' + d1+ '_0'+ d3 + '.npz'
        dest = FP+'/' + i + '.npz'
        F_positive.append(i)
        # shutil.copyfile(file_name, dest)
print('Number of FN = ', len(F_negative))
print('Number of FP = ', len(F_positive))

T_positive=[]
T_negative=[]
for i in correct_vid_id:
    d1 = str(i[0:4])
    d2 = '_1'
    d3 = str(i[5:11])
    d = d1 + d2 + d3
    # print(d)
    # file_name = train_directory +d + '.npz'
    file_name = train_directory +i + '.npz'
    b = np.load(file_name, allow_pickle=True)
    label = b['labels']
    count = 0
    if label[1]> 0:
        dest = TP+'/' + d1+ '_1'+ d3+ '.npz'
        # dest = TP+'/' + i + '.npz'
        T_positive.append(i)
        count+=1
        if count <=61:
            shutil.copyfile(file_name, dest)

    else:
        # dest = TN+'/' + d1+ '_0'+ d3 + '.npz'
        dest = TN+'/' + i + '.npz'
        T_negative.append(i)
        # shutil.copyfile(file_name, dest)
print('Number of TN = ', len(T_negative))
print('Number of TP = ', len(T_positive))

fnr = len(F_negative)/(len(F_negative)+ len(T_positive))
fpr = len(F_positive)/(len(F_positive)+ len(T_negative))
# print('log_prior = %.6f' % (log_prior))
print('FNR = %.2f' %(fnr))
print('FPR = %.2f' %(fpr))
