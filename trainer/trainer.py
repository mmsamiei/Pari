from torch import nn
from torch import optim
class Trainer:
    def __init__(self, model, dataloader, PAD_IDX):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def init_weights(self):
        for name, param in self.model.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        return sum(params)

    def train_one_epoch(self, clip=10):
        self.model.train() #This will turn on dropout (and batch normalization)
        epoch_loss = 0
        for i, batch in enumerate(self.dataloader):
            src, trg, _ = batch
            self.optimizer.zero_grad()
            src = src.permute(1,0)
            trg = trg.permute(1,0)
            # trg = [trg sent len, batch size]
            output = self.model(src, trg)
            # output = [trg sent len, batch size, output dim]
            trg = trg[1:].contiguous().view(-1)
            # trg = [(trg sent len - 1) * batch size]
            output = output[1:].view(-1, output.shape[-1])
            # output = [(trg sent len - 1) * batch size, output dim]
            loss = self.criterion(output, trg)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss

    def train(self, epoch):
        epoch_losses = []
        for i in range(epoch):
            epoch_loss = self.train_one_epoch()
            epoch_losses.append(epoch_loss)
        print(epoch_losses)