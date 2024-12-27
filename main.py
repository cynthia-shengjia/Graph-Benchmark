
import sys

sys.path.append("")

import model
import optimizer
from tools.utils import *
# from logger import Logger


from tools.procedure import train_batch as train_batch


from tools.procedure import test as test
from tools.utils import read_data as read_data
from ogb.linkproppred import Evaluator
import tools.world as world


log_print		= get_logger('testrun', 'log', get_config_dir())


LOSSES = {
    "BCE": optimizer.optim_BCE.BCEOptimizer
}

MODELS = {
    "GCN":  model.model_GCN.GCN,
    "SAGE": model.model_SAGE.SAGE,
    "GAT":  model.model_GAT.GAT
}

SCORE_MODELS = {
    "mlp_score":    model.model_Score.mlp_score
}



args = world.parse_args()
print(args)
init_seed(args.seed)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = read_data("cora", "normal_data", args.filename)

node_num = data['x'].size(0)
x = data['x']
x = x.to(device)

train_pos           = data['train_pos'].to(x.device)
interaction_tensor  = data['interaction_tensor']

input_channel = x.size(1)
model = MODELS[args.gnn_model](input_channel, args.hidden_channels,
                args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, node_num).to(device)
score_func = SCORE_MODELS[args.score_model](args.hidden_channels, args.hidden_channels,
                1, args.num_layers_predictor, args.dropout).to(device)

loss_func = LOSSES[world.config["loss"]](model=model, score_model = score_func, config=world.config)


eval_metric = args.metric
evaluator_hit = Evaluator(name='ogbl-collab')
evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)

loggers = {
    'Hits@1': Logger(args.runs),
    'Hits@3': Logger(args.runs),
    'Hits@10': Logger(args.runs),
    'Hits@100': Logger(args.runs),
    'MRR': Logger(args.runs),

}


run = 0
seed = args.seed
print('seed: ', seed)

init_seed(seed)

save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.weight_decay) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'seed_'+str(seed)



model.reset_parameters()
score_func.reset_parameters()



best_valid = 0
kill_cnt = 0
for epoch in range(1, 1 + args.epochs):
    loss = train_batch(model, score_func, loss_func, train_pos, x, interaction_tensor, args.batch_size)


    if epoch % args.eval_steps == 0:
        results_rank, score_emb = test(model, score_func, data, x, evaluator_hit, evaluator_mrr, args.batch_size)

        for key, result in results_rank.items():
            loggers[key].add_result(run, result)

        if epoch % args.log_steps == 0:
            for key, result in results_rank.items():

                print(key)

                train_hits, valid_hits, test_hits = result


                log_print.info(
                    f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_hits:.2f}%, '
                      f'Valid: {100 * valid_hits:.2f}%, '
                      f'Test: {100 * test_hits:.2f}%')
            print('---')

        best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

        if best_valid_current > best_valid:
            best_valid = best_valid_current
            kill_cnt = 0

            if args.save:
                save_emb(score_emb, save_path)
        else:
            kill_cnt += 1
            if kill_cnt > args.kill_cnt:
                print("Early Stopping!!")
                break

for key in loggers.keys():
    print(key)
    loggers[key].print_statistics(run)


result_all_run = {}
for key in loggers.keys():
    print(key)
    best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()
    if key == eval_metric:
        best_metric_valid_str = best_metric
        best_valid_mean_metric = best_valid_mean
    if key == 'AUC':
        best_auc_valid_str = best_metric
        best_auc_metric = best_valid_mean
    result_all_run[key] = [mean_list, var_list]



best_auc_metric = best_valid_mean_metric
print(best_valid_mean_metric, best_auc_metric, result_all_run)






   