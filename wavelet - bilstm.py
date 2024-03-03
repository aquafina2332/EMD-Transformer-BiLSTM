# BiLSTM和LSTM ：区别只是在bidirectional=False/True和nn.Linear(hidden_size, output_size)/hidden_size*2，因此放在同一个文件中的

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
import pywt


torch.manual_seed(0) 
np.random.seed(0)


input_window = 5 # transformer建立长期依赖的效果差
output_window = 1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  



class LstmRNN(nn.Module):
 
    def __init__(self, input_size=1, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,bias = True,bidirectional=True)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(2*hidden_size, output_size) # 全连接层
        self.init_weights() #这个是定义初始化
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x
        
    def init_weights(self): # 这是原来的哦
        initrange = 0.1
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
    


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
    

    #小波去噪处理
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(series), w.dec_len)
    threshold = 0.04  # Threshold for filtering
    coeffs = pywt.wavedec(series, 'db8', level=maxlev)  # 将信号进行小波分解
    # print(coeffs[0].shape)
    # print(len(coeffs))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波
    series = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

    # training_set_scaled=np.array(training_set_scaled)
    # series=series.reshape(-1,1)
 
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
# output[0].shape # 把前面的改一下

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
    plt.legend(['Prediction', 'Truth'], loc='upper right',fontsize=12)
    plt.grid(True, which='both')
    plt.title('Wavelet-BiLSTM',fontsize=20)
    plt.xlabel('Datetime/hour',fontsize=12)
    plt.ylabel('AQI',fontsize=12)
    plt.savefig('wavelet-bilstm-win%d.png' % input_window)
    plt.close()


    return (total_loss / i) ,rmSE,mAE,mAPE

# %% 模型训练
# 运行100个epoch，每隔10个epoch在测试集上评估一下模型
train_data, val_data, scaler= get_data()
model = LstmRNN().to(device)
criterion = nn.MSELoss()
lr = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
epochs = 100

# epoch = 1
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)


    if (epoch % 100 == 0):
        val_loss, rmSE,mAE,mAPE= plot_and_loss(model, val_data, epoch, scaler) # 这个地方是每十个循环存一次
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
m.to_csv('wavelet-bilstm-%d.csv' % input_window)

# torch.save(model.state_dict(), 'bilstm_model.pt')  # 保存模型参数
