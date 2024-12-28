
import sys

sys.path.append("")

import model
import optimizer
from tools.utils import *
# from logger import Logger

from pprint import pprint

from tools.procedure import train_batch as train_batch


from tools.procedure import test as test
from tools.utils import read_data as read_data
from ogb.linkproppred import Evaluator
import tools.world as world

import nni
import os


if not "NNI_PLATFORM" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = world.config["cuda"]
else:
    optimized_params = nni.get_next_parameter()
    world.config.update(optimized_params)



pprint(world.config)

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
init_seed(world.config['seed'])

device = f'cuda:{world.config["device"]}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = read_data("cora", "normal_data", world.config["filename"])

node_num = data['x'].size(0)
x = data['x']
x = x.to(device)

train_pos           = data['train_pos'].to(x.device)
interaction_tensor  = data['interaction_tensor']

input_channel = x.size(1)
model = MODELS[world.config["gnn_model"]](input_channel,
                world.config["hidden_channels"],
                world.config["hidden_channels"],
                world.config["num_layers"],
                world.config["dropout"],
                world.config["gin_mlp_layer"],
                world.config["gat_head"],
                node_num).to(device)

score_func = SCORE_MODELS[world.config["score_model"]](
    world.config["hidden_channels"],
    world.config["hidden_channels"],
                1,
    world.config["num_layers_predictor"],
    world.config["dropout"]).to(device)

loss_func = LOSSES[world.config["loss"]](model=model, score_model = score_func, config=world.config)


eval_metric = world.config["metric"]
evaluator_hit = Evaluator(name='ogbl-collab')
evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)

loggers = {
    'Hits@1': Logger(args.runs),
    'Hits@5': Logger(args.runs),
    'Hits@10': Logger(args.runs),
    'Hits@20': Logger(args.runs),
    'Hits@100': Logger(args.runs),
    'MRR': Logger(args.runs),
}


run = 0
seed = world.config["seed"]
print('seed: ', seed)

init_seed(seed)

save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.weight_decay) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'seed_'+str(seed)



model.reset_parameters()
score_func.reset_parameters()


best_report_model = None
best_valid = 0
kill_cnt = 0
for epoch in range(world.config["epochs"]):

    if epoch % world.config["eval_steps"] == 0 and epoch != 0:
        results_rank, score_emb = test(model, score_func, data, x, evaluator_hit, evaluator_mrr, world.config["batch_size"])

        for key, result in results_rank.items():
            loggers[key].add_result(run, result)

        for key, result in results_rank.items():
            print(key)
            train_hits, valid_hits, test_hits = result

            log_print.info(
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_hits:.4f}%, '
                  f'Valid: {100 * valid_hits:.4f}%, '
                  f'Test: {100 * test_hits:.4f}%')
        print('---')
        
        valid_current = results_rank


        if "NNI_PLATFORM" in os.environ:
            metric = {
                "MRR": valid_current['MRR'][1] * 100,
                "default": valid_current['Hits@20'][1] * 100,
            }
            nni.report_intermediate_result(metric)


        if valid_current['Hits@20'][1] > best_valid:
            best_report_model = valid_current
            best_valid = valid_current['Hits@20'][1]
            kill_cnt = 0

            if world.config["save"]:
                save_emb(score_emb, save_path)
        else:
            kill_cnt += 1
            print("Patience: {}/5".format(kill_cnt))
            if kill_cnt >= world.config["kill_cnt"]:
                print(best_report_model)
                print("Early Stopping!!")
                break
    
    loss = train_batch(model, score_func, loss_func, train_pos, x, interaction_tensor, world.config["batch_size"])




if "NNI_PLATFORM" in os.environ:
    metric = {"default": best_report_model["Hits@20"][1] * 100, "MRR": best_report_model["MRR"][1] * 100 }
    nni.report_final_result(metric)





   