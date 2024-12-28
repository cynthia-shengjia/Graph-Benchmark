from optimizer.optim_Base import IROptimizer
import torch



class SLOptimizer(IROptimizer):
    def __init__(self, model, config):
        super().__init__()

        # === Model ===
        self.model = model


        # === Hyper-parameter ===
        self.lr = config['lr']
        self.weight_decay = config["weight_decay"]
        self.temp = config["ssm_temp"]

        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam([
            {'params': self.model.parameters(),         'lr': self.lr},
        ], weight_decay=self.weight_decay)


    def cal_loss(self, pos_logits, neg_logits):

        d = neg_logits - pos_logits.unsqueeze(1)


        loss = torch.logsumexp(d / self.temp, dim = -1)
        return loss.mean()


    def step(self, x, pos_edge, neg_edge, adj, perm):

        h           = self.model(x, adj)

        pos_src_emb, pos_dst_emb = h[pos_edge[:,0]], h[pos_edge[:,1]]
        neg_src_emb, neg_dst_emb = h[neg_edge[0]],   h[neg_edge[1]]
        
        pos_logits = (pos_src_emb * pos_dst_emb).sum(dim = -1)
        neg_logits = (neg_src_emb * neg_dst_emb).sum(dim = -1).squeeze()
        


        bce_loss = self.cal_loss(pos_logits, neg_logits)
        loss = bce_loss
        self.optimizer_descent.zero_grad()

        loss.backward()

        self.optimizer_descent.step()
        return bce_loss.cpu().item()

    def save(self, path):
        all_states = self.model.state_dict()
        torch.save(obj=all_states, f=path)

