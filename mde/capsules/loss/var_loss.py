import torch


class VARLoss(torch.nn.Module):
    def __init__(self, label_smoothing=0.0, reduction='none', L=680, use_mse=True, alpha=0.1,  patch_nums=None):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)
        self.loss_weight = torch.ones(1, L) / L

        
        if patch_nums is not None:
            start = 0
            self.loss_weight = torch.ones(1, L)
            for p in patch_nums:
                self.loss_weight[:, start:start + p ** 2] = 1 / (len(patch_nums) * p ** 2)
                start += p ** 2

        if use_mse:
            self.mse_loss = torch.nn.MSELoss()
        self.use_mse = use_mse
        self.alpha = alpha

    def forward(self, batch):
        prog_si = batch.prog_si
        B = batch.image.shape[0]
        self.loss_weight = self.loss_weight.to(batch.image.device)

        loss = self.loss(batch.pred.view(-1, batch.vocab_size), batch.target.view(-1))
        loss = loss.view(B, -1)
        if prog_si >= 0:
            bg, ed = batch.begin_ends[prog_si]
            assert batch.pred.shape[1] == batch.target.shape[1] == ed
            lw = self.loss_weight[:, :ed].clone()
            lw[:, bg:ed] *= min(max(1, 0), 1)
        else:
            lw = self.loss_weight
        loss = loss.mul(lw).sum(dim=-1).mean()
        if self.use_mse:
            mse_loss = 0
            for i in range(len(batch.target_depth_mse[4:])): 
                
                mse_loss += self.mse_loss(batch.pred_depth_mse[4+i].mean(axis=1, keepdims=True), batch.target_depth_mse[4+i].mean(axis=1, keepdims=True))
          #  print(mse_loss)
            
            if loss + mse_loss > 200:
                print(loss, mse_loss)
                np.save('pred', batch.pred_depth.mean(axis=1, keepdims=True).cpu().detach())
                np.save('depth', batch.depth.cpu().detach())
                assert False
            return self.alpha * mse_loss + loss
        return loss
    