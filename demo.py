from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os, sys
import os.path as osp
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        VGG = models.vgg16(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
        self.dim_feat = 4096

    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

def init_feature_extractor(backbone='vgg16', device=torch.device('cuda')):
    feat_extractor = None
    if backbone == 'vgg16':
        feat_extractor = VGG16()
        feat_extractor = feat_extractor.to(device=device)
        feat_extractor.eval()
    else:
        raise NotImplementedError
    return feat_extractor


def bbox_sampling(bbox_result, nbox=19, imsize=None, topN=5):
    """
    imsize[0]: height
    imsize[1]: width
    """
    assert not isinstance(bbox_result, tuple)

    bboxes = np.vstack(bbox_result)  # n x 5

    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]

    labels = np.concatenate(labels)  # n

    ndet = bboxes.shape[0]

    # fix bbox
    new_boxes = []
    for box, label in zip(bboxes, labels):
        x1 = min(max(0, int(box[0])), imsize[1])
        y1 = min(max(0, int(box[1])), imsize[0])
        x2 = min(max(x1 + 1, int(box[2])), imsize[1])
        y2 = min(max(y1 + 1, int(box[3])), imsize[0])
        if (y2 - y1 + 1 > 2) and (x2 - x1 + 1 > 2):
            new_boxes.append([x1, y1, x2, y2, box[4], label])
            # print('box[4] :', box[4])
    if len(new_boxes) == 0:  # no bboxes
        new_boxes.append([0, 0, imsize[1]-1, imsize[0]-1, 1.0, 0])
    new_boxes = np.array(new_boxes, dtype=int)
    # print('new_boxes : ', new_boxes)
    # sampling
    n_candidate = min(topN, len(new_boxes))
    if len(new_boxes) <= nbox - n_candidate:
        indices = np.random.choice(n_candidate, nbox - len(new_boxes), replace=True)
        sampled_boxes = np.vstack((new_boxes, new_boxes[indices]))
    elif len(new_boxes) > nbox - n_candidate and len(new_boxes) <= nbox:
        indices = np.random.choice(n_candidate, nbox - len(new_boxes), replace=False)
        sampled_boxes = np.vstack((new_boxes, new_boxes[indices]))
    else:
        sampled_boxes = new_boxes[:nbox]

    return sampled_boxes


def bbox_to_imroi(transform, bboxes, image):
    imroi_data = []
    for bbox in bboxes:
        imroi = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        imroi = transform(Image.fromarray(imroi))  # (3, 224, 224), torch.Tensor
        imroi_data.append(imroi)
    imroi_data = torch.stack(imroi_data)
    return imroi_data

def extract_features(detector, feat_extractor, video_file, n_frames=100, n_boxes=19):
    assert os.path.join(video_file), video_file
    # prepare video reader and data transformer
    videoReader = mmcv.VideoReader(video_file)
    transform = transforms.Compose([
        # transforms.Resize(256),
        transforms.Resize(512),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )
    features = np.zeros((n_frames, n_boxes + 1, feat_extractor.dim_feat), dtype=np.float32)
    detections = np.zeros((n_frames, n_boxes, 6))  # (50 x 19 x 6)
    frame_prev = None
    for idx in range(n_frames):
        if idx >= len(videoReader):
            print("Copy frame from previous time step.")
            frame = frame_prev.copy()
        else:
            frame = videoReader.get_frame(idx)
        # run object detection inference
        bbox_result = inference_detector(detector, frame)
        # print(idx)

        # print('frame : ', idx ,' shape : ', bbox_result )
        # sampling a fixed number of bboxes
        bboxes = bbox_sampling(bbox_result, nbox=n_boxes, imsize=frame.shape[:2])

        # print('frame : ', idx ,' bboxes : ', bboxes )

        detections[idx, :, :] = bboxes
        # prepare frame data
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            # bboxes to roi feature
            ims_roi = bbox_to_imroi(transform, bboxes, frame)
            ims_roi = ims_roi.float().to(device=device)
            feature_roi = feat_extractor(ims_roi)
            # extract image feature
            ims_frame = transform(Image.fromarray(frame))
            ims_frame = torch.unsqueeze(ims_frame, dim=0).float().to(device=device)
            feature_frame = feat_extractor(ims_frame)
        # obtain feature matrix
        features[idx, 0, :] = np.squeeze(feature_frame.cpu().numpy()) if feature_frame.is_cuda else np.squeeze(feature_frame.detach().numpy())
        features[idx, 1:, :] = np.squeeze(feature_roi.cpu().numpy()) if feature_roi.is_cuda else np.squeeze(feature_roi.detach().numpy())
        frame_prev = frame
    return detections, features


def init_accident_model(model_file, dim_feature=4096, hidden_dim=512, latent_dim=256, n_obj=19, n_frames=50, fps=10.0):
    # building model
    model = UString(dim_feature, hidden_dim, latent_dim,
        n_layers=1, n_obj=n_obj, n_frames=n_frames, fps=fps, with_saa=True, uncertain_ranking=True)
    # model = UString(dim_feature, hidden_dim, latent_dim,
    #     n_layers=1, n_obj=n_obj, n_frames=n_frames, fps=fps, with_saa=True, uncertain_ranking=True, use_mask=False)
    model = model.to(device=device)
    model.eval()
    # load check point
    model, _, _ = load_checkpoint(model, filename=model_file, isTraining=False)
    return model


def load_input_data(feature_file, device=torch.device('cuda')):
    # load feature file and return the transformed data
    data = np.load(feature_file)
    features = data['data']  # 50 x 20 x 4096
    labels = [0, 1]
    detections = data['det']  # 50 x 19 x 6
    toa = [45]  # [useless]

    def generate_st_graph(detections):
        # create graph edges
        num_frames, num_boxes = detections.shape[:2]
        num_edges = int(num_boxes * (num_boxes - 1) / 2)
        graph_edges = []
        edge_weights = np.zeros((num_frames, num_edges), dtype=np.float32)
        for i in range(num_frames):
            # generate graph edges (fully-connected)
            edge = generate_graph_from_list(range(num_boxes))
            graph_edges.append(np.transpose(np.stack(edge).astype(np.int32)))  # 2 x 171
            # compute the edge weights by distance
            edge_weights[i] = compute_graph_edge_weights(detections[i, :, :4], edge)  # 171,
        return graph_edges, edge_weights

    def generate_graph_from_list(L, create_using=None):
        import networkx, itertools
        G = networkx.empty_graph(len(L),create_using)
        if len(L)>1:
            if G.is_directed():
                    edges = itertools.permutations(L,2)
            else:
                    edges = itertools.combinations(L,2)
            G.add_edges_from(edges)
        graph_edges = list(G.edges())

        return graph_edges

    def compute_graph_edge_weights(boxes, edges):
        """
        :param: boxes: (19, 4)
        :param: edges: (171, 2)
        :return: weights: (171,)
        """
        N = boxes.shape[0]
        assert len(edges) == N * (N-1) / 2
        weights = np.ones((len(edges),), dtype=np.float32)
        for i, edge in enumerate(edges):
            c1 = [0.5 * (boxes[edge[0], 0] + boxes[edge[0], 2]),
                0.5 * (boxes[edge[0], 1] + boxes[edge[0], 3])]
            c2 = [0.5 * (boxes[edge[1], 0] + boxes[edge[1], 2]),
                0.5 * (boxes[edge[1], 1] + boxes[edge[1], 3])]
            d = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
            weights[i] = np.exp(-d)

        # normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)  # N*(N-1)/2,
        else:
            weights = np.ones((len(edges),), dtype=np.float32)


        return weights

    graph_edges, edge_weights = generate_st_graph(detections)
    # transform to torch.Tensor
    features = torch.Tensor(np.expand_dims(features, axis=0)).to(device)         #  50 x 20 x 4096
    labels = torch.Tensor(np.expand_dims(labels, axis=0)).to(device)
    graph_edges = torch.Tensor(np.expand_dims(graph_edges, axis=0)).long().to(device)
    edge_weights = torch.Tensor(np.expand_dims(edge_weights, axis=0)).to(device)
    toa = torch.Tensor(np.expand_dims(toa, axis=0)).to(device)
    detections = np.expand_dims(detections, axis=0)
    vid = feature_file.split('/')[-1].split('.')[0]

    return features, labels, graph_edges, edge_weights, toa, detections, vid


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if isTraining:
            optimizer.load_state_dict(checkpoint['optimizer'])
        # print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


def parse_results(all_outputs, batch_size=1, n_frames=50):
    # parse inference results
    pred_score = np.zeros((batch_size, n_frames), dtype=np.float32)
    pred_au = np.zeros((batch_size, n_frames), dtype=np.float32)
    pred_eu = np.zeros((batch_size, n_frames), dtype=np.float32)
    # run inference
    for t in range(n_frames):
        # prediction
        # pred = all_outputs[t]['pred_mean']  # B x 2
        pred = all_outputs[t]  # B x 2
        pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
        pred_score[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
        # uncertainties
        # aleatoric = all_outputs[t]['aleatoric']  # B x 2 x 2
        # aleatoric = aleatoric.cpu().numpy() if aleatoric.is_cuda else aleatoric.detach().numpy()
        # epistemic = all_outputs[t]['epistemic']  # B x 2 x 2
        # epistemic = epistemic.cpu().numpy() if epistemic.is_cuda else epistemic.detach().numpy()
        # pred_au[:, t] = aleatoric[:, 0, 0] + aleatoric[:, 1, 1]
        # pred_eu[:, t] = epistemic[:, 0, 0] + epistemic[:, 1, 1]
        pred_au= 0
        pred_eu = 0
    return pred_score, pred_au, pred_eu


def get_video_frames(video_file, n_frames=50):
    # get the video data
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    counter = 0
    while (ret):
        video_data.append(frame)
        ret, frame = cap.read()
        counter += 1
    assert len(video_data) >= n_frames, video_file
    video_data = video_data[:n_frames]
    return video_data


def preprocess_results(pred_score, aleatoric, epistemic, cumsum=False):
    from scipy.interpolate import make_interp_spline
    std_alea = 1.0 * np.sqrt(aleatoric)
    std_epis = 1.0 * np.sqrt(epistemic)
    # sampling
    xvals = np.linspace(0,len(pred_score)-1,10)
    pred_mean_reduce = pred_score[xvals.astype(np.int)]
    # pred_std_alea_reduce = std_alea[xvals.astype(np.int)]
    # pred_std_epis_reduce = std_epis[xvals.astype(np.int)]
    # smoothing
    xvals_new = np.linspace(1,len(pred_score)+1, p.n_frames)
    pred_score = make_interp_spline(xvals, pred_mean_reduce)(xvals_new)
    # std_alea = make_interp_spline(xvals, pred_std_alea_reduce)(xvals_new)
    # std_epis = make_interp_spline(xvals, pred_std_epis_reduce)(xvals_new)
    pred_score[pred_score >= 1.0] = 1.0-1e-3
    xvals = np.copy(xvals_new)
    # copy the first value into x=0
    xvals = np.insert(xvals_new, 0, 0)
    pred_score = np.insert(pred_score, 0, pred_score[0])
    # std_alea = np.insert(std_alea, 0, std_alea[0])
    # std_epis = np.insert(std_epis, 0, std_epis[0])
    # take cummulative sum of results
    std_alea = 0
    std_epis = 0
    if cumsum:
        pred_score = np.cumsum(pred_score)
        pred_score = pred_score / np.max(pred_score)
    return xvals, pred_score, std_alea, std_epis


def draw_curve(xvals, pred_score, std_alea, std_epis):
    # ax.fill_between(xvals, pred_score - std_alea, pred_score + std_alea, facecolor='wheat', alpha=0.5)
    # ax.fill_between(xvals, pred_score - std_epis, pred_score + std_epis, facecolor='yellow', alpha=0.5)
    pred_score = pred_score *100
    plt.plot(xvals, pred_score, linewidth=3.0)
    plt.axhline(y=0.5, xmin=0, xmax=max(xvals)/(p.n_frames + 2), linewidth=3.0, color='g', linestyle='--')
    # plt.grid(True)
    plt.tight_layout()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='visualize', choices=['extract_feature', 'inference', 'visualize'])
    parser.add_argument('--gpu_id', help='GPU ID', type=int, default=0)
    parser.add_argument('--n_frames', type=int, help='The number of input video frames.', default=100)
    parser.add_argument('--fps', type=float, help='The fps of input video.', default=10.0)
    # feature extraction
    parser.add_argument('--video_file', type=str, default='demo/000821.mp4')
    # parser.add_argument('--mmdetection', type=str, help="the path to the mmdetection.", default="lib/mmdetection")
    parser.add_argument('--mmdetection', type=str, help="the path to the mmdetection.", default="/mmdetection")
    # inference
    parser.add_argument('--feature_file', type=str, help="the path to the feature file.", default="demo/000821_feature.npz")
    parser.add_argument('--ckpt_file', type=str, help="the path to the model file.", default="demo/final_model_ccd.pth")
    # visualize
    parser.add_argument('--result_file', type=str, help="the path to the result file.", default="demo/000821_result.npz")
    parser.add_argument('--vis_file', type=str, help="the path to the visualization file.", default="demo/000821_vis.avi")
    p = parser.parse_args()

    device = torch.device('cuda:'+str(p.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')
    if p.task == 'extract_feature':
        from mmdet.apis import init_detector, inference_detector, show_result
        import mmcv
        # init object detector
        currentDirectory = os.getcwd()
        cfg_file = osp.join(p.mmdetection, "configs/cascade_rcnn_x101_64x4d_fpn_1x_kitti2d.py")
        # cfg_file = osp.join(currentDirectory, "mmdetection/configs/cascade_rcnn_x101_64x4d_fpn_1x_kitti2d.py")
        model_file = osp.join(p.mmdetection, "work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_kitti2d/latest.pth")
        # model_file = osp.join(currentDirectory, "mmdetection/work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_kitti2d/latest.pth")
        #for faster R-cnn
        # cfg_file = osp.join(currentDirectory, "mmdetection/configs/faster_rcnn_x101_64x4d_fpn_1x_coco.py")
        # model_file = osp.join(currentDirectory, "mmdetection/work_dirs/faster_rcnn_x101_64x4d_fpn_1x/faster_rcnn_x101_64x4d_fpn_1x.pth")
        # print('model_file : ', model_file)
        detector = init_detector(cfg_file, model_file, device=device)
        # init feature extractor
        feat_extractor = init_feature_extractor(backbone='vgg16', device=device)
        # object detection & feature extraction
        detections, features = extract_features(detector, feat_extractor, p.video_file, n_frames=p.n_frames)
        feat_file = p.video_file[:-4] + '_feature.npz'
        np.savez_compressed(feat_file, data=features, det=detections)
    elif p.task == 'inference':
        from src.Models_attention_v18_visualization import UString
        # load feature file
        features, labels, graph_edges, edge_weights, toa, detections, vid = load_input_data(p.feature_file, device=device)
        # print('det : ', detections)
        file_name = 'feat_for_check.npz'
        graph_ed = graph_edges.cpu()
        edge_wt = edge_weights.cpu()
        np.savez_compressed(file_name, features=features.cpu(), labels=labels.cpu(), graph_edges=graph_ed, edge_weights=edge_wt, toa= toa.cpu())
        # np.savez_compressed(file_name, graph_edges=graph_ed, edge_weights=edge_wt)

        # prepare model
        model = init_accident_model(p.ckpt_file, dim_feature=features.shape[-1], n_frames=p.n_frames, fps=p.fps)
        with torch.no_grad():
            # run inference
            losses, all_outputs, all_hidden, all_alphas, alphas_ta = model(features, labels, toa, graph_edges, hidden_in=None, edge_weights=edge_weights, npass=10, eval_uncertain=True)
        alphas = all_alphas
        # alphas_ta = np.array(alphas_ta)
        # alphas_ta = alphas_ta.cpu()
        # print('alphas : ', alphas)

        print('-----------------')
        # parse and save results
        pred_score, pred_au, pred_eu = parse_results(all_outputs)
        result_file = osp.join(osp.dirname(p.feature_file), p.feature_file.split('/')[-1].split('_')[0] + '_result.npz')
        np.savez_compressed(result_file, score=pred_score[0], aleatoric=pred_au, epistemic=pred_eu, det=detections[0], alphas = alphas, alphas_ta = alphas_ta)
    elif p.task == 'visualize':
        video_data = get_video_frames(p.video_file, n_frames=p.n_frames)
        all_results = np.load(p.result_file, allow_pickle=True)
        pred_score, aleatoric, epistemic, detections , alphas, alphas_ta= all_results['score'], all_results['aleatoric'], all_results['epistemic'], all_results['det'], all_results['alphas'], all_results['alphas_ta']
        xvals, pred_score, std_alea, std_epis = preprocess_results(pred_score, aleatoric, epistemic, cumsum=False)
        # print('alphas spatial attention',alphas.shape)
        print('======================')
        # print('alphas temporal attention',alphas_ta[11].shape)
        print('+++++++++++++++++++++')

        fig, ax = plt.subplots(1, figsize=(24, 3.5))
        fontsize = 25
        plt.ylim(0, 1.1)
        plt.xlim(0, len(xvals)+1)
        plt.ylabel('Probability', fontsize=fontsize)
        plt.xlabel('Frame (FPS=2)', fontsize=fontsize)
        plt.xticks(range(0, len(xvals)+1, 2), fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        from matplotlib.animation import FFMpegWriter
        curve_writer = FFMpegWriter(fps=2, metadata=dict(title='Movie Test', artist='Matplotlib',comment='Movie support!'))
        curve_save = p.video_file[:-4] + '_curve_video.mp4'
        with curve_writer.saving(fig, curve_save, 100):
            for t in range(len(xvals)):
                # draw_curve(xvals[:(t+1)], pred_score[:(t+1)], std_alea[:(t+1)], std_epis[:(t+1)])
                draw_curve(xvals[:(t+1)], pred_score[:(t+1)], 0, 0)
                curve_writer.grab_frame()
        curve_frames = get_video_frames(curve_save, n_frames=p.n_frames)

        # create video writer
        video_writer = cv2.VideoWriter(p.vis_file, cv2.VideoWriter_fourcc(*'DIVX'), 2.0, (video_data[0].shape[1], video_data[0].shape[0]))
        for t, frame in enumerate(video_data):
            attention_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.uint8)

            now_weight = alphas[t]
            # print('now_weight shape : ', now_weight.shape)
            # alphas_ta_weight =  alphas_ta[t]
            # alpha_new = torch.softmax(torch.sum(alphas_ta_weight,-1),0)
            # print('alphas_ta_weight', alpha_new.shape)
            # print(alphas_ta_weight.shape)
            # print('now_weight max :', max(now_weight)*6)
            # print('===================================')
            # if t>9:
            #     print('alphas_ta_weight max :', torch.max(alphas_ta_weight), 'for t ', t)
            now_weight = now_weight.cpu()
            now_weight = now_weight*255
            det_boxes = detections[t]  # 19 x 6
            index = np.argsort(now_weight)
            for num_box in index:
            #     # print(now_weight[num_box])
                # if now_weight[num_box]/255.0>0.0007:
                #     # cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
                #     cv2.rectangle(frame,(int(det_boxes[num_box,0]),int(det_boxes[num_box,1])),(int(det_boxes[num_box,2]),int(det_boxes[num_box,3])),(0,255,0),3)
                # else:
                #     # cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                #     cv2.rectangle(frame,(int(det_boxes[num_box,0]),int(det_boxes[num_box,1])),(int(det_boxes[num_box,2]),int(det_boxes[num_box,3])),(50,0,0),3)
                # attention_frame[int(new_bboxes[num_box,1]):int(new_bboxes[num_box,3]),int(new_bboxes[num_box,0]):int(new_bboxes[num_box,2])] = now_weight[num_box]
                # attention_frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = now_weight[num_box]
                attention_frame[int(det_boxes[num_box,1]):int(det_boxes[num_box,3]),int(det_boxes[num_box,0]):int(det_boxes[num_box,2])] = now_weight[num_box]*50000
                # attention_frame[:,:] = now_weight[num_box]*50000
            # height = int(img.shape[0] * (width / img.shape[1]))
            attention_frame_resized = cv2.resize(attention_frame,(frame.shape[1], frame.shape[0]))
            attention_frame = cv2.applyColorMap(attention_frame_resized, cv2.COLORMAP_BONE)
            # attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)

            # dst = cv2.addWeighted(frame,0.9,attention_frame,0.4,0)
            dst = cv2.addWeighted(frame,0.9,attention_frame,0.4,0)
            img = curve_frames[t]
            width = frame.shape[1]
            height = int(img.shape[0] * (width / img.shape[1]))
            img = cv2.resize(dst, (width, height), interpolation = cv2.INTER_AREA)
            # dst = cv2.addWeighted(frame,1,attention_frame,0.3,0)
            # frame[frame.shape[0]-height:frame.shape[0]] = cv2.addWeighted(frame[frame.shape[0]-height:frame.shape[0]],0.6,attention_frame,0.4,0)
            video_writer.write(dst)


            # for box in det_boxes:
            #     # print(box)
            #     if box[4] > 0.0001:
            #         print(box[4])
            #     cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
            # img = curve_frames[t]
            # width = frame.shape[1]
            # height = int(img.shape[0] * (width / img.shape[1]))
            # img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
            # frame[frame.shape[0]-height:frame.shape[0]] = cv2.addWeighted(frame[frame.shape[0]-height:frame.shape[0]], 0.3, img, 0.7, 0)
            # video_writer.write(frame)
    else:
        print("invalid task.")
