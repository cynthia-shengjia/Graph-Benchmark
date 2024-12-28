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
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device="cuda"), torch.zeros(
            neg_logits.shape, device="cuda"
        )

        logits = torch.cat((pos_logits, neg_logits), dim=-1)
        labels = torch.cat((pos_labels, neg_labels), dim=-1)

        bce_criterion = torch.nn.BCEWithLogitsLoss()
        loss = bce_criterion(logits, labels)
        return loss


    def step(self, x, pos_edge, neg_edge, adj, perm):

        h           = self.model(x, adj)

        pos_src_emb, pos_dst_emb = h[pos_edge[:,0]], h[pos_edge[:,1]]
        neg_src_emb, neg_dst_emb = h[neg_edge[0]],   h[neg_edge[1]]
        
        pos_logits = (pos_src_emb * pos_dst_emb).sum(dim = -1)
        neg_logits = (neg_src_emb * neg_dst_emb).sum(dim = -1).squeeze()
        

        # pos_logits = self.score_model(pos_src_emb,pos_dst_emb).squeeze()
        # neg_logits = self.score_model(neg_src_emb, neg_dst_emb).squeeze()


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

