from torch import nn
import torch
from torch import optim
import time
import os
import datetime


class Trainer:
    def __init__(self, model, dataloader, validation_dataloader, PAD_IDX, dev):
        self.model = model
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.dev = dev

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
            src = src.to(self.dev)
            trg = trg.to(self.dev)
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

    def train(self, N_epoch, save_period = 10):
        epoch_losses = []
        valid_losses = []
        directory_name = datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')
        os.makedirs('./saved_models/{}'.format(directory_name))
        for i_epoch in range(N_epoch):
            start_time = time.time()
            epoch_loss = self.train_one_epoch()
            epoch_losses.append(epoch_loss)
            valid_losses.append(self.evaluate())
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            print("epoch {}, loss is {}".format(i_epoch,epoch_loss))
            if(i_epoch % save_period == 0):
                temp_path = os.path.join('.','saved_models')
                temp_path = os.path.join(temp_path,directory_name)
                temp_path = os.path.join(temp_path, 'model-{}.pt'.format(i_epoch))
                torch.save(self.model.state_dict(), temp_path)
            ## TODO

        print(epoch_losses)

    def evaluate(self):
        self.model.eval() #This will turn off dropout (and batch normalization)
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(self.validation_dataloader):
                src, trg, _ = batch
                src = src.to(self.dev)
                trg = trg.to(self.dev)
                src = src.permute(1, 0)
                trg = trg.permute(1, 0)
                output = self.model(src, trg, 0)
                trg = trg[1:].contiguous().view(-1)
                output = output[1:].view(-1, output.shape[-1])
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        return  epoch_loss/len(self.validation_dataloader)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs



