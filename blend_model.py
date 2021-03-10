import torch
import copy

import dnnlib
import legacy
import pickle


def blend(low_res_pkl, high_res_pkl, save_pkl, res):
    # Load networks.
    print('Loading base networks from "%s"...' % low_res_pkl)
    with dnnlib.util.open_url(low_res_pkl) as fp:
        tmp = legacy.load_network_pkl(fp)
        low_res_Gs = tmp['G_ema'].requires_grad_(False).cuda()

    print('Loading refined networks from "%s"...' % high_res_pkl)
    with dnnlib.util.open_url(high_res_pkl) as fp:
        high_res_Gs = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).cuda()

    res_list = ['b'+str(i) for i in [4, 8, 16, 32, 64, 128, 256] if i <= res ]

    for (name, para) in low_res_Gs.named_parameters():
        # print("before, ", torch.equal(low_res_Gs.state_dict()[name].data, high_res_Gs.state_dict()[name].data))
        if name.split('.')[1] in res_list:
            para.copy_(high_res_Gs.state_dict()[name].data)

        # print("after, ", torch.equal(low_res_Gs.state_dict()[name].data, high_res_Gs.state_dict()[name].data))

    
    tmp['G_ema'] = copy.deepcopy(low_res_Gs).eval().requires_grad_(False).cpu()

    with open(save_pkl+str(res)+'.pkl', 'wb') as f:
        pickle.dump(tmp, f)


if __name__ == '__main__':

    low_res_pkl = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl'
    high_res_pkl = 'result/00000-256X256-mirror-paper256-kimg40-target0.5-custom-resumecelebahq256-freezed10/network-snapshot-000040.pkl'
    save_pkl = 'model_blend'
    for res in [4,8,16,32,64,128,256]:
        blend(low_res_pkl, high_res_pkl, save_pkl, res)