import numpy as np


def NMS(preds, overlapThresh=0.2):
    boxes = []
    for pred in preds:
        boxes.append(pred['bbox'])

    indices = np.arange(len(boxes))
    for i, box in enumerate(boxes):
        for j in indices[indices != i]:
            overlap = bb_intersection_over_union(box, boxes[j])
            if overlap > overlapThresh:
                discard = j if preds[i]['score'] > preds[j]['score'] else i
                indices = indices[indices != discard]

    final_pred = []
    for k in list(indices):
        final_pred.append(preds[k])
    return final_pred


def bb_intersection_over_union(box1, box2):
    """determine the (x, y)-coordinates of the intersection rectangle (XYXY Mode for bbox)"""
    # xywh mode to xyxy mode
    x, y, w, h = box1
    boxA = [x, y, x + w, y + h]
    x, y, w, h = box2
    boxB = [x, y, x + w, y + h]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def preds_sort(pred_json_ls):
    # prediction processing
    #     pred_dict_ls = []
    pred_dict = {}
    for pred_json in pred_json_ls:
        for item in pred_json:
            if item['image_id'] in pred_dict.keys():
                pred_dict[item['image_id']].append(item)
            else:
                pred_dict[item['image_id']] = [item]

    return pred_dict


def bagging(pred_dicts, nms_th):
    org_sum = 0
    final_preds = []
    imgIDs = list(pred_dicts.keys())
    for imgID in imgIDs:
        preds = pred_dicts[imgID]
        org_sum += len(preds)

        if len(preds) >= 2:
            final_preds += NMS(preds, nms_th)
        else:
            final_preds += preds
    print('%.3f percent is kept after nms filtering' % (len(final_preds) / org_sum * 100))

    return final_preds


def retina_bags(pred_jsons):
    # resort pred_json based on img_id
    pred_dicts_sorted = []
    org_num = 0
    for pred_json in pred_jsons:
        org_num += len(pred_json)
        pred_dicts_sorted.append(preds_sort([pred_json]))

    imgIds = [set(pred_dict.keys()) for pred_dict in pred_dicts_sorted]
    imgIds_u = set({})
    for imgId in imgIds:
        imgIds_u = imgIds_u.union(imgId)

    # filter proposals
    final_preds = []
    for img_id in list(imgIds_u):
        preds_current_img = [pred_dict[img_id] for pred_dict in pred_dicts_sorted if img_id in pred_dict.keys()]

        pred_toeval = preds_current_img[0]
        for i in np.arange(1, len(preds_current_img)):
            for pred_1 in pred_toeval:
                for pred_2 in preds_current_img[i]:
                    iou = bb_intersection_over_union(pred_1['bbox'], pred_2['bbox'])
                    if iou >= 0.2:
                        final_preds.append(pred_1)
                        # discard collected proposal in waiting pool
                        #                         import pdb; pdb.set_trace()
                        pred_toeval.remove(pred_1)
                        break

            pred_toeval = preds_current_img[i]

    print('%.3f percent is kept after retina bag filtering' % (len(final_preds) / org_num * 100))

    return final_preds

