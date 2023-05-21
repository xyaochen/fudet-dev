import mmcv
import torch
import numpy as np
'''
def isok(box):
    mask = (box[0] > -51.2 and
            box[0] < 51.2 and
            box[1] > -51.2 and 
            box[1] < 51.2 and 
            box[2] > -3 and
            box[2] < 5)

    return mask

path = 'data/nusc_new/nuscenes_infos_train.pkl'
infos = mmcv.load(path)
#import ipdb; ipdb.set_trace()

infos = infos['infos']
x = []
y = []
z = []
w = []
l = []
h = []
cnt = 0
for info in infos:
    gt_boxes = info['gt_boxes']
    for gt_box in gt_boxes:
        if isok(gt_box):
            x.append(gt_box[0])
            y.append(gt_box[1])
            z.append(gt_box[2])
            w.append(gt_box[3])
            l.append(gt_box[4])
            h.append(gt_box[5])
            cnt += 1
    if cnt % 1000 == 0:
        print(cnt)
        

x = np.array(x)
y = np.array(y) 
z = np.array(z)
w = np.log(np.array(w))
l = np.log(np.array(l))
h = np.log(np.array(h))

print('x', x.mean(), x.max(), x.min())
print('y', y.mean(), y.max(), y.min())
print('z', z.mean(), z.max(), z.min())
print('w', w.mean(), w.max(), w.min())
print('l', l.mean(), l.max(), l.min())
print('h', h.mean(), h.max(), h.min())

cls_cnt = \
    {0: 121948, 1: 98870, 2: 43698, 3: 46305, 4: 42712, 5: 57220, \
            6: 34818, 7: 35273, 8: 104225, 9: 65612}

s = sum([v for k,v in cls_cnt.items()])
ratios = [v/s for k,v in cls_cnt.items()]
for ratio in ratios:
    print(6*0.1 / ratio)

pre = len('pts_bbox_head.transformer.decoder.layers.0.attentions.1.')
path = '../fusion-det/pretrained/detr3d_res101.pth'

ckpt = torch.load(path)['state_dict']
new_sd = {}
for name in ckpt:
    if 'output_proj' in name:
        new_name = name[:pre] + 'img_output_proj.'
        if 'bias' in name:
            new_name += 'bias'
        else:
            new_name += 'weight'
        new_sd[new_name] = ckpt[name]
    elif 'attention_weights' in name:
        new_name = name[:pre] + 'img_attention_weights.'
        if 'bias' in name:
            new_name += 'bias'
        else:
            new_name += 'weight'
        new_sd[new_name] = ckpt[name]
    else:
        new_sd[name] = ckpt[name]

for name in new_sd:
    print(name)

new_ckpt = {'state_dict': new_sd}

torch.save(new_ckpt, 'checkpoint/detr3d_converted.pth')

path = '../futr_lidar_cam.pth'
torch.load(path)
'''
path = '../nuscenes_infos_train.pkl'
infos = mmcv.load(path)
truncated_infos = {}
truncated_infos['metadata'] = infos['metadata']
truncated_infos['infos'] = infos['infos'][:N]

mmcv.dump(truncated_infos, '../truncated_infos_train.pkl')

import ipdb; ipdb.set_trace()