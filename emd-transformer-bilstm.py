#%% 探索性分析
# import pandas as pd
# import pandas_profiling # 探索性可视化
# df = pd.read_csv('air_patna.csv', usecols=['AQI'])
# pandas_profiling.ProfileReport(df) 

# 最终的目标模型
# %% 导入库和相关的参数设置
import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from PyEMD import EMD
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
# from tensorboardX import SummaryWriter  # 用于进行可视化
# from torchviz import make_dot
# logger = SummaryWriter(log_dir="data/log")
# writer_g = SummaryWriter("data/generator") 

torch.manual_seed(0) 
np.random.seed(0)

input_window = 5 # 更改参数
output_window = 1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  


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
class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.hidden_size = 2
        self.decoder = nn.LSTM(input_size=feature_size,hidden_size=self.hidden_size ,num_layers=2,bias = True,bidirectional=True)
        self.linear1 = nn.Linear(self.hidden_size*2, 1) # 全连接层
        self.init_weights2()
        for name, param in self.decoder.named_parameters(): 
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
 
    def init_weights2(self): 
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
def get_data(series):


    train_data = series[:train_samples]
    test_data = series[train_samples:]
 
  
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]
 
 
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]
 
 
    return train_sequence.to(device), test_data.to(device)

def get_batch(source, i, batch_size): # 便于以batch形式读取
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target

# %% 模型构建
# 对参数进行反向传播，其中用到了梯度裁剪的技巧用于防止梯度爆炸
def train(train_data,model):
    model.train() # 查看结构

    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        start_time = time.time()
        total_loss = 0
        data, targets = get_batch(train_data, i, batch_size) # 看看输入
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets) # lstm的输出有两个
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'
                  .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_last_lr()[0], elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
    # # 添加的第一条日志：损失函数-全局迭代次数
    # logger.add_scalar("loss", loss.item(), global_step=epoch)
    # logger.add_scalar("lr", scheduler.get_last_lr()[0] ,global_step=epoch)
            
# %% 模型评估
def evaluate(eval_model, data_source):
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    eval_model.eval() 
    total_loss = 0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size): # 运行一遍
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)  
            truth = torch.cat((truth, targets[-1].view(-1).cpu()), 0)

    test_result = (test_result.reshape(-1,1)).detach().numpy()
    # truth = (truth.reshape(-1,1)).detach().numpy()
    # rmSE = metrics.mean_squared_error(test_result, truth)**0.5   
    # mAE = metrics.mean_absolute_error(truth,test_result)
    # mAPE = metrics.mean_absolute_percentage_error(truth,test_result)
    
    # # 添加第二条日志：RMSE-全局迭代次数
    # logger.add_scalar("RMSE", rmSE, global_step=epoch)
    # logger.add_scalar("MAE", mAE, global_step=epoch)
    # logger.add_scalar("MAPE", mAPE, global_step=epoch)
    
    return  test_result

# %% 线性回归
def linearm(data):
    open_arr =data.reshape(-1, 1).reshape(-1) # 读数据
    X = np.zeros(shape=(len(open_arr) - input_window, input_window))
    label = np.zeros(shape=(len(open_arr) - input_window))
    for i in range(len(open_arr) - input_window):
        X[i, :] = open_arr[i:i+input_window]
        label[i] = open_arr[i+input_window]
    train_X = X[:train_samples, :]
    train_label = label[:train_samples]
    test_X = X[train_samples:, :]
    test_label = label[train_samples:]
    
    linreg = LinearRegression()
    model=linreg.fit(train_X,train_label)
    y_pred = linreg.predict(test_X)
    
    return  y_pred ,test_label

# %% 模型训练
# 运行100个epoch，每隔10个epoch在测试集上评估一下模型
# series = pd.read_csv('air_patna.csv', usecols=['AQI'])

series = pd.read_csv('air_patna.csv', usecols=['AQI'])
scaler = MinMaxScaler(feature_range=(-1, 1))
series = scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)
train_samples = int(round(len(series)*0.75,0))
# train_samples = 30800
# emd_win = 24
emd = EMD()
IMFs = emd(series,max_imf=12) # EMD 分解
# from pyhht.visualization import plot_imfs
# plot_imfs(series, np.array(IMFs))
# pd.DataFrame(IMFs).T.to_csv('imf.csv')

N = len(IMFs)
k=0
for itm in IMFs:
    k=k+1
    globals()['series'+str(k)] = itm
    globals()['train_data'+str(k)],globals()['val_data'+str(k)] = get_data(locals()['series'+str(k)])

# 写一个for循环把每个模型都训练好
lr = 0.001
epochs = 100
# j = 0
for j in range(len(IMFs)):
    j = j+1
    globals()['model'+str(j)] = TransAm().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(locals()['model'+str(j)].parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    # epoch = 1
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(locals()['train_data'+str(j)],locals()['model'+str(j)]) # model
        scheduler.step()


# 如果线性回归效果更好就用线性回归
pred = []
# i=0
for i in range(len(IMFs)):
    i = i+1
    globals()['res'+str(i)] = evaluate(locals()['model'+str(i)],locals()['val_data'+str(i)])
    reslinear,ttru = linearm(IMFs[i-1])
    ttru = ttru[2-len(ttru):]
    reslinear = reslinear[2-len(reslinear):]
    if metrics.mean_squared_error(reslinear, ttru)**0.5 < metrics.mean_squared_error(locals()['res'+str(i)], ttru)**0.5 :
        locals()['res'+str(i)] = reslinear.tolist()
        pred.append(locals()['res'+str(i)])
    else:
        pred.append([token for st in locals()['res'+str(i)] for token in st])
        
# # 分别得到loss
# data = torch.randn(2, 2, 1).to(device) # 定义一个网络的输入值,形状相似的
# logger.add_graph(model, data)
# # 保存成pt文件后进行可视化
# torch.save(model, "../log/modelviz.pt")
# # 使用graphviz进行可视化
# out = model(data)
# g = make_dot(out)
# g.render('modelviz', view=False)
# logger.close()


# %% y~IMFS
# 多元BILSTM对时间进行预测
class LstmRNN(nn.Module): # BILSTM
    
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,bidirectional=True)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(2*hidden_size, output_size) # 全连接层
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x
    
i = 0
x = []
for item in IMFs:
    i = i+1
    globals()['train'+str(i)] = item[:train_samples]
    x.append(locals()['train'+str(i)])
X = pd.DataFrame(x).T  # 检查下这个
data_x = np.array(X).astype('float32')
train_data = series[:train_samples]
data_y = np.array(train_data).astype('float32')

data_len = len(data_x)
t = np.linspace(0, data_len, data_len)

train_x = data_x
train_y = data_y
t_for_training = t

INPUT_FEATURES_NUM = len(IMFs)
OUTPUT_FEATURES_NUM = 1
lstm_model = LstmRNN(INPUT_FEATURES_NUM, 20, output_size=OUTPUT_FEATURES_NUM, num_layers=1)  # 20 hidden units
lstm_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  
train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM) 
train_x_tensor = torch.from_numpy(train_x_tensor)
train_y_tensor = torch.from_numpy(train_y_tensor)
train_x_tensor = train_x_tensor.to(device)
train_y_tensor = train_y_tensor.to(device) 

epoches = 1000
for epoch in range(epoches):  
    output = lstm_model(train_x_tensor).to(device)
    loss = criterion(output, train_y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  
# %% 预测值与真实值-结果评价
pred = np.array(pred).astype('float32')
test_x = np.transpose(pred) # IMFs的series 测试集
# test_x = pred0 # IMFs的series 测试集
test_x_tensor = test_x.reshape(-1, 1, INPUT_FEATURES_NUM)  
test_x_tensor = torch.from_numpy(test_x_tensor)
test_x_tensor = test_x_tensor.to(device)
pre = lstm_model(test_x_tensor).to(device)
pre = (pre.cpu().reshape(-1,1)).detach().numpy().flatten()
test_y = series[-len(pre):]
pre = scaler.inverse_transform(pre.reshape(-1, 1))
test_y = scaler.inverse_transform(test_y.reshape(-1, 1)) # 反归一化

rmSE = metrics.mean_squared_error(pre, test_y)**0.5   
mAE = metrics.mean_absolute_error(pre,test_y)
mAPE = metrics.mean_absolute_percentage_error(pre,test_y) # 计算指标

# pre = series[-7720:]
# test_y = pre
plt.figure(figsize=(10,5))
plt.plot(pre, color="red")
plt.plot(test_y, color="blue")
plt.legend(['Prediction', 'Truth'], loc='upper right',fontsize=12)
plt.grid(True, which='both') 
plt.title('EMD-Transformer-BiLSTM',fontsize=20)
plt.xlabel('Datetime/hour',fontsize=12)
plt.ylabel('AQI',fontsize=12)
# plt.savefig('mm.png')
plt.savefig('emd-transformer-bilstm-win%d.png' % input_window)
plt.close()

m = []
m.append(rmSE)
m.append(mAE)
m.append(mAPE)
m = pd.DataFrame(m)
m.to_csv('emd-tr-bilstm-%d-h.csv' % input_window)

# # %% 画一下模型的损失图 ，tensorboard画的太丑了
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# lossdf = pd.read_csv('loss3.csv')
# plt.plot(lossdf['test_loss'], color="cornflowerblue")
# plt.plot(lossdf['train_loss'], color="sandybrown")
# plt.yscale('log') # 切换对数刻度
# plt.legend(['test_loss', 'train_loss'], loc='upper right',fontsize=13)
# # plt.grid(True, which='both')
# plt.title('Test Loss and Train Loss-IMF3',fontsize=15) #写上图题
# plt.xlabel('Epochs',fontsize=13) #为x轴命名
# plt.ylabel('Loss',fontsize=13) #为y轴命名
# plt.savefig('loss3.png')
# plt.close()