import torch
import collections

if __name__ == "__main__":
    # 加载原权重文件
    weights_files_all = '/home/zhaoyibin/3DRE/sfm-learn/SFM_OWN/feature/LightGlue/glue-factory/outputs/training/sp+lg_homography_save/checkpoint_best.tar'
    weights_all = torch.load(weights_files_all)
    weights_files_ok = '/home/zhaoyibin/3DRE/sfm-learn/SFM_OWN/feature/LightGlue/wight/superpoint_lightglue.pth'
    weights_ok = torch.load(weights_files_ok)
    # 修改
    for cen in weights_all['model'].keys():
        print('自己训练的：',cen)
        print(weights_all['model'][cen].shape)
        print(weights_ok[cen].shape)




    new_d = weights_all
    new_d['model'] = weights_ok
    # 保存
    torch.save(new_d, 'trans_weight.tar')