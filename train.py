import json
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.nn as nn
import torch.optim as optim
from Model import Transformer
import time
import os
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def load_input(path,angle, length):
    source_file= path + angle + '/source.json'
    target_file= path +angle + '/target.json'
    with open(source_file, 'r') as sf:
        with open(target_file, 'r') as tf:
            sl = json.load(sf)[length]
            tl= json.load(tf)[length]
        tf.close()
    sf.close()
    return sl,tl

def convert_to_Longtensor(data):
    data = torch.LongTensor(data)
    return data

def make_decoderinput(data):
    for i in range(len(data)):
        data[i].insert(0, 1001)
        # data[i].append(36002)
    return data

def recover_target(data):
    for i in range(len(data)):
        del data[i][0]
    return data

def make_decoderoutput(data):
    for i in range(len(data)):
        data[i].append(1002)
    return data


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

# split train data to train_sampler, valid_sampler（7：3）
def dataset_split(dataset_size, random_seed):
    split1 = .7
    indices = list(range(dataset_size))
    train_split = int(np.floor(split1 * dataset_size))
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[: train_split], indices[train_split:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    return train_sampler, valid_sampler

def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    t_loss = 0.
    start_time = time.time()
    for step, (enc_inputs, dec_inputs, dec_outputs) in enumerate(train_loader):
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        # print(outputs,len(outputs),len(outputs[0]))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        t_loss += loss.item()
        log_interval = 100
        if step % log_interval == 0:
            if( step != 0 ):
                cur_loss = total_loss / log_interval
                elapsed = (time.time() - start_time) / log_interval
            else:
                cur_loss = total_loss
                elapsed = time.time() - start_time
            print('| epoch {:3d} | batches {:5d} | '
                  'lr {:0.5f} | s/batch {:5.2f} | '
                  'loss {:7.4f} '.format(epoch, step, lr_scheduler.get_last_lr()[0], elapsed , cur_loss))
            total_loss = 0
            start_time = time.time()

    return t_loss / len(train_loader)


def evaluate(eval_model, psi_valid_loader):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for enc_inputs, dec_inputs, dec_outputs in psi_valid_loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # print(criterion(outputs, dec_outputs.view(-1)).item())
            total_loss += criterion(outputs, dec_outputs.view(-1)).item()
    return total_loss / len(psi_valid_loader)


if __name__ == "__main__":

    train_path = "data/source/train/"
    test_path1= "data/source/test/CASP14/"
    test_path2 = "data/source/test/CASP15/"
    angles = ["psi", "phi", "omega", "CCN", "CNC", "NCC"]
    minlength,maxlength = 5,37
    # choose angle

    for angle in angles:

        train_batchsize = 64

        train_source, train_target = load_input(train_path,angle)
        print('train_data')
        print(len(train_source),len(train_target))

        train_enc_inputs = convert_to_Longtensor(train_source)
        train_dec_inputs = convert_to_Longtensor(make_decoderinput(train_target))
        train_dec_outputs = convert_to_Longtensor(make_decoderoutput(recover_target(train_target)))
        print('train_tensor_data')
        # print(train_enc_inputs)
        # print(train_dec_inputs)
        # print(train_dec_outputs)
        print(len(train_enc_inputs))
        print(len(train_dec_inputs))
        print(len(train_dec_outputs))

        # generate train_sampler, valid_sampler
        dataset = MyDataSet(train_enc_inputs, train_dec_inputs, train_dec_outputs)
        train_sampler, valid_sampler = dataset_split(len(dataset), random_seed=9)
        train_loader = DataLoader(dataset, train_batchsize, shuffle=False, sampler=train_sampler)
        valid_loader = DataLoader(dataset, train_batchsize, shuffle=False, sampler=valid_sampler)


        print('\n-----Training Helix--%s-----' % angle)

        print('----------TRAIN-----------')
        RESUME = False
        EPOCH = 200
        start_epoch = 0
        modeldir = "AnglesRefine/model/helix/%s_model" % angle
        best_val_loss = float("inf")
        plotPath = "AnglesRefine/model/helix/plot/%s_plot/loss/" % angle

        if RESUME:
            path_checkpoint = "%s/ckpt_best.pth" % modeldir
            checkpoint = torch.load(path_checkpoint)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
            loss = checkpoint['loss']
            best_val_loss = loss

        # ----------------------------------------tensorboard----------------------------------------
        loss_show = []
        trainPlotPath = plotPath + "train"
        validPlotPath = plotPath + "valid"
        train_writer = SummaryWriter(trainPlotPath)
        val_writer = SummaryWriter(validPlotPath)
        # ----------------------------------------tensorboard----------------------------------------

        for epoch in range(start_epoch + 1, EPOCH + 1):
            epoch_start_time = time.time()
            train_loss = train()
            val_loss = evaluate(model, valid_loader)


            train_loss = float('%.4f' % train_loss)
            val_loss = float('%.4f' % val_loss)

            # ----------------------------------------tensorboard----------------------------------------
            train_writer.add_scalar("loss", train_loss, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            # ----------------------------------------tensorboard----------------------------------------

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:7.4f} | valid loss {:7.4f} |  '
                  .format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))
            print('-' * 89)
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "lr_schedule": lr_scheduler.state_dict(),
                "loss": val_loss
            }

            if not os.path.exists(modeldir):
                os.makedirs(modeldir)
            lr_scheduler.step()
            val_loss = float('%.4f' % val_loss)
            best_val_loss = float('%.4f' % best_val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                torch.save(checkpoint, '%s/ckpt_best.pth' % modeldir)

        best_path_checkpoint = '%s/ckpt_best.pth' % modeldir
        bestCheckpoint = torch.load(best_path_checkpoint)
        print('\n-----Helix-%s-----' % angle)
        print('\nBEST EPOCH:    LOSS    ', bestCheckpoint['epoch'], ':    ', bestCheckpoint['loss'])









