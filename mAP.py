import sys
import pylab as pl

if len(sys.argv) != 3:
    print('usage python mAP.py $PREDICT_FILE $GROUND_TRUTH_FILE\n')
    exit()

# sys.argv[0]表示程序代码本身的路径
predict_file = sys.argv[1]
ground_truth_file = sys.argv[2]
predict_dict = dict()
ground_truth_dict = dict()

def get_info(info_file, info_dict):
    bbox_num = 0
    first_line = True
    with open(info_file) as fr:

        for line in fr:

            if first_line:
                first_line = False
                continue

            if len(line.strip().split(',')) ==6:
                class_id, xmin, ymin, xmax, ymax, conf = map(float,  line.strip().split(','))
            else:
                class_id, xmin, ymin, xmax, ymax = map(float,  line.strip().split(','))
                conf = 1.

            if not class_id in info_dict:
                info_dict[class_id] = list()

            info_dict[class_id].append([xmin, ymin, xmax, ymax, conf])
            bbox_num += 1

    return bbox_num

predict_bbox_num = get_info(predict_file, predict_dict)
ground_truth_bbox_num = get_info(ground_truth_file, ground_truth_dict)
conf_list = list()
match_list = list()

def iou(predict_bbox, ground_truth_bbox):
    predict_area = (predict_bbox[2] - predict_bbox[0]) * (predict_bbox[3] - predict_bbox[1])
    ground_truth_area = (ground_truth_bbox[2] - ground_truth_bbox[0]) * (ground_truth_bbox[3] - ground_truth_bbox[1])
    inter_x = min(predict_bbox[2], ground_truth_bbox[2]) - max(predict_bbox[0], ground_truth_bbox[0])
    inter_y = min(predict_bbox[3], ground_truth_bbox[3]) - max(predict_bbox[1], ground_truth_bbox[1])
    if inter_x <= 0 or inter_y <= 0:
        return 0
    inter_area = inter_x * inter_y

    return inter_area / (predict_area + ground_truth_area - inter_area)

def compare(predict_list, ground_truth_list, conf_list, match_list):
    ground_truth_unuse = [True for i in range(len(ground_truth_list))]

    for predict_bbox in predict_list:

        match = False
        for i in range(len(ground_truth_list)):
            if ground_truth_unuse[i]:
                if iou(predict_bbox, ground_truth_list[i]) > 0.5:
                    match = True
                    ground_truth_unuse[i] = False
                    break

        conf_list.append(predict_bbox[-1])
        match_list.append(int(match))

for key in predict_dict.keys():
    compare(predict_dict[key], ground_truth_dict[key], conf_list, match_list)

p = list()
r = list()
predict_num = 0
truth_num = 0
conf_match_list = list(zip(conf_list, match_list))
conf_match_list.sort(key = lambda x: x[0], reverse = True)

for item in conf_match_list:
    predict_num += 1
    truth_num += item[1]
    p.append(float(truth_num) / ground_truth_bbox_num)
    r.append(float(truth_num) / predict_num)

mAP = 0
for i in range(1, len(p)):
    mAP += (r[i-1] + r[i]) / 2 * (p[i] - p[i-1])
print('mAP:{}'.format(mAP))
pl.plot(p, r)
pl.show()





























































