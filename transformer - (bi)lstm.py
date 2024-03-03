# Tranformer-BiLSTM

# %% 导入库和相关的参数设置
import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import metrics


torch.manual_seed(0) 
np.random.seed(0)


input_window = 3 # transformer建立长期依赖的效果差
output_window = 1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # 悲报：没有服务器


# %% 模型构建1： 位置编码
class PositionalEncoding(nn.Module):
 
 
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
 
 
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
# %%  transforme框架
# 没有采用原论文中的Encoder-Decoder的架构，而是将Decoder用了一个全连接层进行代替，用于输出预测值。
# 另外，其中的create_mask将输入进行mask，从而避免引入未来信息
class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.decoder = nn.Linear(feature_size, 1)
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=10,dropout=dropout)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.hidden_size = 2
        self.decoder = nn.LSTM(input_size=feature_size,hidden_size=self.hidden_size ,num_layers=2,bias = True,bidirectional=False)
#         self.init_weights() #这个是定义初始化 
        self.linear1 = nn.Linear(self.hidden_size, 1) # 全连接层
        self.init_weights2()

        for name, param in self.decoder.named_parameters(): # 这个是可以运行的哦
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

#     def init_weights(self):  # 这个是lstm的stvd分解
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#           weight.data.uniform_(-stdv, stdv)
 
    def init_weights2(self): # 这是原来的哦
        initrange = 0.1
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output, _ = self.decoder(output)
        s, b, h = output.shape 
        output = output.view(s * b, h)
        output = self.linear1(output)
        output = output.view(s, b, -1)
        return output
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    


# %% 数据预处理：滑窗处理
# 假设输入是1到20，则其标签就是2到21，以适应Transformer的seq2seq的形式的输出
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)

# %% 数据分割与get
def get_data():


    series = pd.read_csv('air_patna.csv', usecols=['AQI'])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    series = scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)
 
 
    train_samples = int(round(len(series)*0.75,0))
    train_data = series[:train_samples]
    test_data = series[train_samples:]
 
  
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]
 
 
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]
 
 
    return train_sequence.to(device), test_data.to(device), scaler

def get_batch(source, i, batch_size): # 便于以batch形式读取
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target

# %% 模型构建
# 对参数进行反向传播，其中用到了梯度裁剪的技巧用于防止梯度爆炸
def train(train_data):
    model.train() # 查看结构
 
# bilstm = nn.LSTM(input_size=1, hidden_size=1, num_layers=2, bidirectional = False)
# output, (hn, cn) =bilstm(data)
# output.shape # 把前面的改一下

    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        start_time = time.time()
        total_loss = 0
        data, targets = get_batch(train_data, i, batch_size) # 看看输入
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets) # lstm的输出有两个
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'
                  .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_last_lr()[0], elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            
# %% 模型评估 
def evaluate(eval_model, data_source):
    eval_model.eval() 
    total_loss = 0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)

# %% 模型结果绘图
def plot_and_loss(eval_model, data_source, epoch, scaler):
    eval_model.eval()
    # model.eval()
    # data_source = val_data
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)   # 这个要返回
            # output,(hn,cn) = model(data)
            total_loss += criterion(output, target).item() # output[0]怎么是两列
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)  # 这个地方检查一下linear的output[0][-1]结构
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    test_result = (test_result.reshape(-1,1)).detach().numpy()
    truth = (truth.reshape(-1,1)).detach().numpy()
    test_result = scaler.inverse_transform(test_result)
    truth = scaler.inverse_transform(truth) # 反归一化
    rmSE = metrics.mean_squared_error(test_result, truth)**0.5   # 这个地方错误
    mAE = metrics.mean_absolute_error(truth,test_result)
    mAPE = metrics.mean_absolute_percentage_error(truth,test_result)
    
    plt.figure(figsize=(10,5))
    plt.plot(test_result, color="red")
    plt.plot(truth, color="blue")
    plt.legend(['Prediction', 'Truth'], loc='upper right')
    plt.grid(True, which='both')
    plt.title('Transformer-LSTM')
    plt.xlabel('Datetime')
    plt.ylabel('AQI')
    plt.savefig('transformer-lstm-epoch%d.png' % epoch)
    plt.close()

    return (total_loss / i) ,rmSE,mAE,mAPE

# %% 模型训练
# 运行100个epoch，每隔10个epoch在测试集上评估一下模型
train_data, val_data, scaler= get_data()
model = TransAm().to(device)
criterion = nn.MSELoss()
lr = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
epochs = 100


# epoch = 1
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)


    if (epoch % 50 == 0):
        val_loss,rmSE,mAE,mAPE= plot_and_loss(model, val_data, epoch, scaler) # 这个地方是每十个循环存一次
    else:
        val_loss = evaluate(model, val_data)


    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    # print(rmSE)
    print('-' * 89)
    scheduler.step()

print(rmSE)
print(mAE)
print(mAPE)

m = []

m.append(rmSE)
m.append(mAE)
m.append(mAPE)
m = pd.DataFrame(m)
m.to_csv('tr-lstm-%d.csv' % input_window)

# torch.save(model.state_dict(), 'tran_bilstm1_model.pt')  # 保存模型参数

# #读取
# model = TransAm().to(device)
# model.load_state_dict(torch.load('tran_bilstm_model.pt'))

