# coding: utf-8
"""
Pengyi Zhang
201906
"""
import cv2

import argparse
import json
import os

import numpy

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *
from utils.parse_config import *

""" Slim Principle
(1) Use global threshold to control pruning ratio
(2) Use local threshold to keep at least 10% unpruned 
"""

def route_conv(layer_index, module_defs):
    """ find the convolutional layers connected by route layer
    """
    module_def = module_defs[layer_index]
    mtype = module_def['type']
    
    before_conv_id = []
    if mtype in ['convolutional', 'shortcut', 'upsample', 'maxpool']:
        if module_defs[layer_index-1]['type'] == 'convolutional':
            return [layer_index-1]
        before_conv_id += route_conv(layer_index-1, module_defs)

    elif mtype == "route":
        layer_is = [int(x)+layer_index if int(x) < 0 else int(x) for x in module_defs[layer_index]['layers'].split(',')]
        for layer_i in layer_is: 
            if module_defs[layer_i]['type'] == 'convolutional':
                before_conv_id += [layer_i]
            else:
                before_conv_id += route_conv(layer_i, module_defs)
        
    return before_conv_id


def write_model_cfg(old_path, new_path, new_module_defs):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    lines = []
    with open(old_path, 'r') as fp:
        old_lines = fp.readlines()
    for _line in old_lines:
        if "[convolutional]" in _line:
            break
        lines.append(_line)

    for i, module_def in enumerate(new_module_defs):
        
        mtype = module_def['type']
        lines.append("[{}]\n".format(mtype))
        print("layer:", i, mtype)
        if mtype == "convolutional":
            bn = 0
            filters = module_def['filters']
            bn = int(module_def['batch_normalize'])
            if bn:
                lines.append("batch_normalize={}\n".format(bn))
                filters = torch.sum(module_def['mask']).cpu().numpy().astype('int')        
            lines.append("filters={}\n".format(filters))
            lines.append("size={}\n".format(module_def['size']))
            lines.append("stride={}\n".format(module_def['stride']))
            lines.append("pad={}\n".format(module_def['pad']))
            lines.append("activation={}\n\n".format(module_def['activation']))
        elif mtype == "shortcut":
            lines.append("from={}\n".format(module_def['from']))
            lines.append("activation={}\n\n".format(module_def['activation']))   
        elif mtype == 'route':
            lines.append("layers={}\n\n".format(module_def['layers']))               
            
        elif mtype == 'upsample':
            lines.append("stride={}\n\n".format(module_def['stride']))
        elif mtype == 'maxpool':
            lines.append("stride={}\n".format(module_def['stride']))
            lines.append("size={}\n\n".format(module_def['size']))
        elif mtype == 'yolo':
            lines.append("mask = {}\n".format(module_def['mask']))
            lines.append("anchors = {}\n".format(module_def['anchors']))
            lines.append("classes = {}\n".format(module_def['classes']))
            lines.append("num = {}\n".format(module_def['num']))
            lines.append("jitter = {}\n".format(module_def['jitter']))
            lines.append("ignore_thresh = {}\n".format(module_def['ignore_thresh']))
            lines.append("truth_thresh = {}\n".format(module_def['truth_thresh']))
            lines.append("random = {}\n\n".format(module_def['random']))
    
    with open(new_path, "w") as f:
        f.writelines(lines)


        
def test(
        cfg,
        weights=None,
        img_size=406,
        save=None,
        overall_ratio=0.5,
        perlayer_ratio=0.1
):

    """prune yolov3 and generate cfg, weights
    """
    if save != None:
        if not os.path.exists(save):
            os.makedirs(save)
    device = torch_utils.select_device()
    # Initialize model
    model = Darknet(cfg, img_size).to(device)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        _state_dict = torch.load(weights, map_location=device)['model']
        model.load_state_dict(_state_dict)
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

##  output a new cfg file
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0] # channels numbers
    
    bn = torch.zeros(total)
    index = 0

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    sorted_bn, sorted_index = torch.sort(bn)
    thresh_index = int(total*overall_ratio)
    thresh = sorted_bn[thresh_index].cuda()

    print("--"*30)
    print()
    #print(list(model.modules()))
    # 
    proned_module_defs = model.module_defs
    for i, (module_def, module) in enumerate(zip(model.module_defs, model.module_list)):
        print("layer:", i)
        mtype = module_def['type']
        if mtype  == 'convolutional':
            bn = int(module_def['batch_normalize'])
            if bn:
                m = getattr(module, 'batch_norm_%d' % i) # batch_norm layer
                weight_copy = m.weight.data.abs().clone()
                channels = weight_copy.shape[0] #
                min_channel_num = int(channels * perlayer_ratio) if int(channels * perlayer_ratio) > 0 else 1
                mask = weight_copy.gt(thresh).float().cuda()  
                
                if int(torch.sum(mask)) < min_channel_num: 
                    _, sorted_index_weights = torch.sort(weight_copy,descending=True)
                    mask[sorted_index_weights[:min_channel_num]]=1. 

                proned_module_defs[i]['mask'] = mask.clone()

                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                        format(i, mask.shape[0], int(torch.sum(mask)))) 

            print("layer:", mtype)

        elif mtype in ['upsample', 'maxpool']:
            print("layer:", mtype)

        elif mtype == 'route':
            print("layer:", mtype)
            # 

        elif mtype == 'shortcut':
            layer_i = int(module_def['from'])+i
            print("from layer ", layer_i)
            print("layer:", mtype)
            proned_module_defs[i]['is_access'] = False
            

        elif mtype == 'yolo':
            print("layer:", mtype)
            

    layer_number = len(proned_module_defs)
    for i in range(layer_number-1, -1, -1):
        mtype = proned_module_defs[i]['type']
        if mtype == 'shortcut': 
            if proned_module_defs[i]['is_access']: 
                continue

            Merge_masks =  []
            layer_i = i
            while mtype == 'shortcut':
                proned_module_defs[layer_i]['is_access'] = True

                if proned_module_defs[layer_i-1]['type'] == 'convolutional': 
                    bn = int(proned_module_defs[layer_i-1]['batch_normalize'])
                    if bn: 
                        Merge_masks.append(proned_module_defs[layer_i-1]["mask"].unsqueeze(0))

                layer_i = int(proned_module_defs[layer_i]['from'])+layer_i 
                mtype = proned_module_defs[layer_i]['type']

                if mtype == 'convolutional':              
                    bn = int(proned_module_defs[layer_i]['batch_normalize'])
                    if bn: 
                        Merge_masks.append(proned_module_defs[layer_i]["mask"].unsqueeze(0))
                

            if len(Merge_masks) > 1:
                Merge_masks = torch.cat(Merge_masks, 0)
                merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float().cuda()
            else:
                merge_mask = Merge_masks[0].float().cuda()

            layer_i = i
            mtype = 'shortcut'
            while mtype == 'shortcut':

                if proned_module_defs[layer_i-1]['type'] == 'convolutional': 
                    bn = int(proned_module_defs[layer_i-1]['batch_normalize'])
                    if bn:
                        proned_module_defs[layer_i-1]["mask"] = merge_mask

                layer_i = int(proned_module_defs[layer_i]['from'])+layer_i 
                mtype = proned_module_defs[layer_i]['type']

                if mtype == 'convolutional': 
                    bn = int(proned_module_defs[layer_i]['batch_normalize'])
                    if bn:     
                        proned_module_defs[layer_i]["mask"] = merge_mask



    for i, (module_def, module) in enumerate(zip(model.module_defs, model.module_list)):
        print("layer:", i)
        mtype = module_def['type']
        if mtype  == 'convolutional':
            bn = int(module_def['batch_normalize'])
            if bn:

                layer_i_1 = i - 1
                proned_module_defs[i]['mask_before'] = None

                mask_before = []
                conv_indexs = []
                if i > 0:
                    conv_indexs = route_conv(i, proned_module_defs)
                    for conv_index in conv_indexs:
                        mask_before += proned_module_defs[conv_index]["mask"].clone().cpu().numpy().tolist()
                    proned_module_defs[i]['mask_before'] = torch.tensor(mask_before).float().cuda()  
                   
                            

 
    output_cfg_path = os.path.join(save, "prune.cfg")
    write_model_cfg(cfg, output_cfg_path, proned_module_defs)

    pruned_model = Darknet(output_cfg_path, img_size).to(device)
    print(list(pruned_model.modules()))
    for i, (module_def, old_module, new_module) in enumerate(zip(proned_module_defs, model.module_list, pruned_model.module_list)):  
        mtype = module_def['type']
        print("layer: ",i, mtype)
        if mtype  == 'convolutional': # 
            bn = int(module_def['batch_normalize'])
            if bn:
                new_norm = getattr(new_module, 'batch_norm_%d' % i) # batch_norm layer
                old_norm = getattr(old_module, 'batch_norm_%d' % i) # batch_norm layer

                new_conv = getattr(new_module, 'conv_%d' % i) # conv layer
                old_conv = getattr(old_module, 'conv_%d' % i) # conv layer  
                

                idx1 = np.squeeze(np.argwhere(np.asarray(module_def['mask'].cpu().numpy())))
                if i > 0:
                    idx2 = np.squeeze(np.argwhere(np.asarray(module_def['mask_before'].cpu().numpy())))
                    new_conv.weight.data = old_conv.weight.data[idx1.tolist()][:, idx2.tolist(), :, :].clone()
                    
                    print("idx1: ", len(idx1), ", idx2: ", len(idx2))
                else:
                    new_conv.weight.data = old_conv.weight.data[idx1.tolist()].clone()

                new_norm.weight.data = old_norm.weight.data[idx1.tolist()].clone()
                new_norm.bias.data = old_norm.bias.data[idx1.tolist()].clone()
                new_norm.running_mean = old_norm.running_mean[idx1.tolist()].clone()
                new_norm.running_var = old_norm.running_var[idx1.tolist()].clone()
                

                print('layer index: ', i, 'idx1: ', idx1)     
            else: 

                new_conv = getattr(new_module, 'conv_%d' % i) # batch_norm layer
                old_conv = getattr(old_module, 'conv_%d' % i) # batch_norm layer
                idx2 = np.squeeze(np.argwhere(np.asarray(proned_module_defs[i-1]['mask'].cpu().numpy())))
                new_conv.weight.data = old_conv.weight.data[:,idx2.tolist(),:,:].clone()
                new_conv.bias.data = old_conv.bias.data.clone()
                print('layer index: ', i, "entire copy") 

    print('--'*30)
    print('prune done!')    
    print('pruned ratio %.3f'%overall_ratio)
    prune_weights_path = os.path.join(save, "prune.pt")    
    _pruned_state_dict = pruned_model.state_dict()
    torch.save(_pruned_state_dict, prune_weights_path)

    print("Done!") 



    # test
    pruned_model.eval()
    img_path = "test.jpg"
    
    org_img = cv2.imread(img_path)  # BGR
    img, ratiow, ratioh, padw, padh = letterbox(org_img, new_shape=[img_size,img_size], mode='rect')

    # Normalize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    imgs = torch.from_numpy(img).unsqueeze(0).to(device)
    _, _, height, width = imgs.shape  # batch size, channels, height, width

    # Run model
    inf_out, train_out = pruned_model(imgs)  # inference and training outputs
    # Run NMS
    output = non_max_suppression(inf_out, conf_thres=0.005, nms_thres=0.5)
    # Statistics per image
    for si, pred in enumerate(output):
        if pred is None:
            continue
        if True:
            box = pred[:, :4].clone()  # xyxy
            scale_coords(imgs[si].shape[1:], box, org_img.shape[:2])  # to original shape
            for di, d in enumerate(pred):
                category_id = int(d[6])
                left, top, right, bot = [float(x) for x in box[di]]
                confidence = float(d[4])

                cv2.rectangle(org_img, (int(left), int(top)), (int(right), int(bot)),
                                (255, 0, 0), 2)
                cv2.putText(org_img, str(category_id) + ":" + str('%.1f' % (float(confidence) * 100)) + "%", (int(left), int(top) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)  
        cv2.imshow("result", org_img)
        cv2.waitKey(-1)            
        cv2.imwrite('result_{}'.format(img_path), org_img)


    # convert pt to weights:
    prune_c_weights_path = os.path.join(save, "prune.weights")
    save_weights(pruned_model, prune_c_weights_path)
    

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='PyTorch Slimming Yolov3 prune')
    parser.add_argument('--cfg', type=str, default='VisDrone2019/yolov3-spp3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='yolov3-spp3_final.weights', help='path to weights file')
    parser.add_argument('--img_size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--save', default='prune', type=str, metavar='PATH', help='path to save pruned model (default: none)')
    parser.add_argument('--overall_ratio', type=float, default=0.5, help='scale sparse rate (default: 0.5)')    
    parser.add_argument('--perlayer_ratio', type=float, default=0.1, help='minimal scale sparse rate (default: 0.1) to prevent disconnect')    
    
    opt = parser.parse_args()
    opt.save += "_{}_{}".format(opt.overall_ratio, opt.perlayer_ratio)

    print(opt)

    with torch.no_grad():
        test(
            opt.cfg,
            opt.weights,
            opt.img_size,
            opt.save,
            opt.overall_ratio,
            opt.perlayer_ratio,
        )
