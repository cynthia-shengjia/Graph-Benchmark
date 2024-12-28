from optimizer.optim_Base import IROptimizer
import torch
from torch import nn



class TopKOptimizer(IROptimizer):
    def __init__(self, model, node_num, config):
        super().__init__()

        # === Model ===
        self.model = model
        self.node_num = node_num
        self.sampled_neg = config["neg_num"]


        # === Hyper-parameter ===
        self.lr = config['lr']
        self.weight_decay = config["weight_decay"]
        self.temp = config["ssm_temp"]
        self.lr2  = config["lr2"]
        self.lambda_t = float(config["lambda_k"] / self.node_num) 


        # === Model Optimizer ===
        self.quantile =nn.Parameter( (torch.zeros((self.node_num, 1))).cuda() )
        self.activation = lambda x: torch.log(torch.sigmoid(x))


        self.optimizer_descent = torch.optim.Adam([
            {'params': self.model.parameters(),         'lr': self.lr},
            {'params': self.quantile, "lr": self.lr2}
        ], weight_decay=self.weight_decay)



    def cal_loss(self, pos_logits, src_neg, dst_neg, src_quantile, dst_quantile):
        trunc_pos1 = pos_logits - src_quantile.squeeze()
        trunc_neg1 = src_neg    - src_quantile

        
        trunc_pos2 = pos_logits - dst_quantile.squeeze()
        trunc_neg2 = dst_neg    - dst_quantile


        pos_part1 = self.activation(trunc_pos1)   / self.temp
        neg_part1 = torch.logsumexp( self.activation(trunc_neg1 )  / self.temp, dim = 1     )

        pos_part2 = self.activation(trunc_pos2)   / self.temp
        neg_part2 = torch.logsumexp( self.activation(trunc_neg2 )  / self.temp, dim = 1     )

        loss1 = neg_part1 - pos_part1
        loss2 = neg_part2 - pos_part2
        loss = loss1 / 2 + loss2 / 2
        
        return loss.mean()

    def quantile_regression(self, h_pad, p1_src, p1_dst, p2_src, p2_dst, src_neg, dst_neg, src_quantile, dst_quantile):


        p1_src_emb = h_pad[p1_src].detach()
        p1_dst_emb = h_pad[p1_dst].detach()
        


        p2_src_emb = h_pad[p2_src].detach()
        p2_dst_emb = h_pad[p2_dst].detach()

        p1_pos_scores = (p1_src_emb * p1_dst_emb).sum(dim = -1)
        p2_pos_scores = (p2_src_emb * p2_dst_emb).sum(dim = -1)

        check_sum1 = (p1_dst != self.node_num)
        check_sum2 = (p2_src != self.node_num)

        trunc_pos1 = (  (1 - self.lambda_t) * torch.relu(p1_pos_scores - src_quantile) + self.lambda_t * torch.relu(src_quantile - p1_pos_scores)  ) * check_sum1
        trunc_neg1 = (1 - self.lambda_t) * torch.relu(src_neg - src_quantile) + self.lambda_t * torch.relu(src_quantile - src_neg)

 
        trunc_pos2 = (  (1 - self.lambda_t) * torch.relu(p2_pos_scores - src_quantile) + self.lambda_t * torch.relu(src_quantile - p2_pos_scores)  ) * check_sum2
        trunc_neg2 = (1 - self.lambda_t) * torch.relu(dst_neg - src_quantile) + self.lambda_t * torch.relu(src_quantile - dst_neg)
 

        weight1 = (self.node_num - check_sum1.sum(-1)).unsqueeze(1) / int( self.sampled_neg / 2 )
        weight2 = (self.node_num - check_sum2.sum(-1)).unsqueeze(1) / int( self.sampled_neg / 2 )


        all_scores1      = (torch.cat((trunc_pos1, weight1 * trunc_neg1), dim = 1)).sum(dim = 1) / self.node_num
        all_scores2      = (torch.cat((trunc_pos2, weight2 * trunc_neg2), dim = 1)).sum(dim = 1) / self.node_num
        
        return all_scores1.mean() / 2 + all_scores2.mean() / 2
        




    def step(self, x, pos_edge, neg_edge, all_pos, adj, perm):

        h           = self.model(x, adj)


        pos_src_emb, pos_dst_emb = h[pos_edge[:,0]], h[pos_edge[:,1]]
        neg_src_emb, neg_dst_emb = h[neg_edge[0]],   h[neg_edge[1]]
        



        src_quantile = self.quantile[pos_edge[:,0]]
        dst_quantile = self.quantile[pos_edge[:,1]]


        
        pos_logits = (pos_src_emb * pos_dst_emb).sum(dim = -1)
        neg_logits = (neg_src_emb * neg_dst_emb).sum(dim = -1).squeeze()

        
        tensor_chunks = torch.chunk(neg_logits, 2, dim=1)

        src_neg = tensor_chunks[0]
        dst_neg = tensor_chunks[1]



        all_pos_part_one = all_pos[0]
        all_pos_part_two = all_pos[1]

        p1_src, p1_dst = all_pos_part_one[0], all_pos_part_one[1]
        p2_src, p2_dst = all_pos_part_two[0], all_pos_part_two[1]
        h_pad = torch.cat( ( h, torch.zeros(1, h.shape[1]).cuda()) )   # padding
        

        
        quantile_loss = quantile_regression = self.quantile_regression(h_pad, p1_src, p1_dst, p2_src, p2_dst, src_neg.detach(), dst_neg.detach(), src_quantile, dst_quantile)

        topk_loss = self.cal_loss(pos_logits, src_neg, dst_neg, src_quantile.detach(), dst_quantile.detach())


        loss = topk_loss + quantile_loss

        self.optimizer_descent.zero_grad()

        loss.backward()

        self.optimizer_descent.step()
        return topk_loss.cpu().item()

    def save(self, path):
        all_states = self.model.state_dict()
        torch.save(obj=all_states, f=path)

