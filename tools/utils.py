import os
import torch
import numpy as np
import random
import logging, sys
import math
import logging.config 
import scipy.sparse as ssp
from torch_sparse import SparseTensor


def get_root_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "..")


def get_config_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "config")


def init_seed(seed=2024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
        
def save_model(model, save_path, emb=None):

    if emb == None:
        state = {
            'state_dict_model'	: model.state_dict(),
            # 'state_dict_predictor'	: linkPredictor.state_dict(),
        }

    else:
        state = {
            'state_dict_model'	: model.state_dict(),
            'emb'	: emb.weight
        }

    torch.save(state, save_path)

def save_emb(score_emb, save_path):

    if len(score_emb) == 6:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x1, x2= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x1,
        'node_emb_with_valid_edges': x2

        }
        
    elif len(score_emb) == 5:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x
        }
    elif len(score_emb) == 4:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, = score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        }
   
    torch.save(state, save_path)

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []

            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')

            r = best_result[:, 0].float()
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1].float()
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 2].float()
            best_train_mean = round(r.mean().item(), 2)
            best_train_var = round(r.std().item(), 2)
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 3].float()
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            var_list = [best_train_var, best_valid_var, best_test_var]


            return best_valid, best_valid_mean, mean_list, var_list



def get_logger(name, log_dir, config_dir):
	
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def read_data(data_name, dir_path, filename, config):
    data_name = data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []

    for split in ['train', 'test', 'valid']:

        path = dir_path + '/{}/{}_pos.txt'.format(data_name, split)

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)

            node_set.add(sub)
            node_set.add(obj)

            if sub == obj:
                continue

            if split == 'train':
                train_pos.append((sub, obj))

            if split == 'valid': valid_pos.append((sub, obj))
            if split == 'test': test_pos.append((sub, obj))

    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge, train_edge[[1, 0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    with open(f'{dir_path}/{data_name}/heart_valid_{filename}', "rb") as f:
        valid_neg = np.load(f)
        valid_neg = torch.from_numpy(valid_neg)
    with open(f'{dir_path}/{data_name}/heart_test_{filename}', "rb") as f:
        test_neg = np.load(f)
        test_neg = torch.from_numpy(test_neg)

    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])

    train_pos_tensor = torch.tensor(train_pos)
    if config['loss'] == 'PreAtK':
        interaction_tensor, user_all_pos = create_interaction_tensor(train_pos,num_nodes)
        interaction_tensor = interaction_tensor.cuda()
        user_all_pos       = user_all_pos.cuda()
    else:
        interaction_tensor, _ = create_interaction_tensor(train_pos,num_nodes)
        interaction_tensor = interaction_tensor.cuda()

    valid_pos = torch.tensor(valid_pos)

    test_pos = torch.tensor(test_pos)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    feature_embeddings = torch.load(dir_path + '/{}/{}'.format(data_name, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {}
    data['adj'] = adj
    data['train_pos'] = train_pos_tensor
    data['train_val'] = train_val

    data['valid_pos'] = valid_pos
    data['valid_neg'] = valid_neg
    data['test_pos'] = test_pos
    data['test_neg'] = test_neg

    data['x'] = feature_embeddings
    data["interaction_tensor"] = interaction_tensor
    
    if config["loss"] == "PreAtK":
        data["user_all_pos"]       = user_all_pos

    return data


def create_interaction_tensor(train_pos, node_num):
    interaction_tensor = torch.eye(node_num, node_num, dtype=torch.bool)
    for x in train_pos:
        interaction_tensor[x[0], x[1]] = 1
        interaction_tensor[x[1], x[0]] = 1
    user_pos_items = create_all_pos_tensor(interaction_tensor, node_num)
    return interaction_tensor, user_pos_items

def create_all_pos_tensor(interaction_tensor, node_num):
    padding_number = node_num
    user_pos_items  = torch.empty((node_num, node_num), dtype = torch.int64)


    for index in range(node_num):
        x = interaction_tensor[index]
        pos_list       =  torch.nonzero(x, as_tuple=True)[0].tolist()
        generate_list  =  pos_list + [node_num] * (node_num - len(pos_list))
        user_pos_items[index] = torch.tensor(generate_list)
    

    return user_pos_items