import torch
def find_key(all_keys, search_key):
    for key in all_keys:
        if search_key in key:
            return key
a = torch.load('pretrained_models/psp_ffhq_frontalization.pt')
b = torch.load('pretrained_models/e4e_encoder.pt')
for new_key in b['state_dict'].keys():
    b['state_dict'][new_key] = a['state_dict'][find_key(a['state_dict'].keys(), new_key)]
torch.save(b,'front_encoder.pt')
