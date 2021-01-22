import os


import torch

PATH = './net_states/'


def get_latest_nets():
    files = list_nets()
    if 'tgt_net' in files[0]:
        tgt_net = files[0]
        net = files[1]
    elif 'tgt_net' in files[1]:
        tgt_net = files[1]
        net = files[0]
    else:
        pass
    return PATH+net, PATH+tgt_net


def list_nets():
    files = os.listdir(PATH)
    return files


def get_net_iter():
    files = list_nets()
    if len(files) > 0:
        iteration = int(files[0].split('_')[-2])
    else:
        iteration = -999
    return iteration, files


def save_network(net, name):
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    torch.save(net.cpu().state_dict(), PATH + name)


def update_actors(net, tgt_net):
    iteration, files = get_net_iter()
    if iteration != -999:
        iteration += 1
        new_net_name = 'net_' + str(iteration) + '_.pt'
        new_tgt_net_name = 'tgt_net_' + str(iteration) + '_.pt'
        for file in files:
            os.remove(PATH + file)
        save_network(net, new_net_name)
        save_network(tgt_net, new_tgt_net_name)


def net_update_check(i):
    iteration, _ = get_net_iter()
    return iteration == i
