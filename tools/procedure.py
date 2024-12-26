from torch.utils.data import DataLoader
import torch
from torch_sparse import SparseTensor
from tools.evalutors import evaluate_mrr


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred,
                     neg_test_pred):
    result = {}
    k_list = [1, 3, 10, 100]

    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred)
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred)

    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in [1, 3, 10, 100]:
        result[f'Hits@{K}'] = (
        result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result



def train_batch(model, score_func, loss_func, train_pos, x, batch_size):
    # 一个 batch 的 train
    model.train()
    score_func.train()
    total_batch = train_pos.size(0) // batch_size + 1
    aver_loss = 0.0


    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):

        num_nodes = x.size(0)

        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0

        train_edge_mask = train_pos[mask].transpose(1, 0)

        # train_edge_mask = to_undirected(train_edge_mask)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1, 0]]), dim=1)
        # edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)

        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(
            train_pos.device)


        # Just do some trivial random sampling.
        pos_edge = train_pos[perm].t()
        neg_edge = torch.randint(0, num_nodes, pos_edge.size(), dtype=torch.long,
                             device="cuda")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)


        loss = loss_func.step(x = x, pos_edge = pos_edge, neg_edge = neg_edge, adj = adj, perm = perm)
        aver_loss += loss


    return aver_loss / total_batch




@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size, negative_data=None):
    pos_preds = []
    neg_preds = []

    if negative_data is not None:
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))

            pos_scores = score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu()
            neg_scores = score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu()

            pos_preds += [pos_scores]
            neg_preds += [neg_scores]

        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = None
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            pos_preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]

    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds, neg_preds


@torch.no_grad()
def test(model, score_func, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()

    h = model(x, data['adj'].to(x.device))
    # print(h[0][:10])
    x = h

    pos_train_pred, _ = test_edge(score_func, data['train_val'], h, batch_size)
    pos_valid_pred, neg_valid_pred = test_edge(score_func, data['valid_pos'], h, batch_size,
                                               negative_data=data['valid_neg'])
    pos_test_pred, neg_test_pred = test_edge(score_func, data['test_pos'], h, batch_size,
                                             negative_data=data['test_neg'])

    pos_train_pred = torch.flatten(pos_train_pred)
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)

    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(),
          neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())

    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred,
                              pos_test_pred, neg_test_pred)

    score_emb = [pos_valid_pred.cpu(), neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]

    return result, score_emb

