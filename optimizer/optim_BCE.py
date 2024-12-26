from optimizer.optim_Base import IROptimizer
import torch


class BCEOptimizer(IROptimizer):
    def __init__(self, model, score_model, config):
        super().__init__()

        # === Model ===
        self.model = model
        self.score_model = score_model

        # === Hyper-parameter ===
        self.lr = config['lr']
        self.weight_decay = config["weight_decay"]

        # === Model Optimizer ===
        self.optimizer_descent = torch.optim.Adam([
            {'params': self.model.parameters(),         'lr': self.lr},
            {'params': self.score_model.parameters(),   'lr': self.lr},
        ], weight_decay=self.weight_decay)


    def cal_loss(self, pos_logits, neg_logits):

        pos_loss    = -torch.log(pos_logits + 1e-15).mean()
        neg_loss    = -torch.log(neg_logits + 1e-15).mean()
        loss        = pos_loss + neg_loss
        return loss.mean()


    def step(self, x, pos_edge, neg_edge, adj, perm):

        h           = self.model(x, adj)
        pos_logits  = self.score_model(h[pos_edge[0]], h[pos_edge[1]])
        neg_logits  = self.score_model(h[neg_edge[0]], h[neg_edge[1]])


        bce_loss = self.cal_loss(pos_logits, neg_logits)
        loss = bce_loss
        self.optimizer_descent.zero_grad()

        loss.backward()

        self.optimizer_descent.step()
        return bce_loss.cpu().item()

    def save(self, path):
        all_states = self.model.state_dict()
        all_states.update({
            "score_model": self.score_model.state_dict()
        })
        torch.save(obj=all_states, f=path)

