import os
import cv2
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

# 最大影響變數
LR = 0.005
BATCH_SIZE = 32
EPOCH = 10
val_size = 0.1
# Data classes 辨識數量
#classes = ('1', '6')
classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
classes_len = len(classes)

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' #選擇gpu或cpu
print('GPU cuda:', torch.cuda.is_available())
print('GPU name:',torch.cuda.get_device_name(0))

def default_loader(path):
    transform = transforms.Compose([
        transforms.Resize([480, 480]),
        transforms.ToTensor(), #0-255 -> 0.0-1.0
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
    img = Image.open(path)
    I = transform(img)
    return I
class MyDataset(Data.Dataset):
    def __init__(self, place, loader=default_loader):
        imgs = []
        #資料的label
        for textpath in os.listdir(place):
            #資料的檔案名字
            for fn in os.listdir(os.path.join(place, textpath)):
                #檔案的相對路徑
                path = os.path.join(place, textpath, fn)
                #加入檔案的路徑與答案
                imgs.append((path, int(textpath))) #img , label
        self.imgs = imgs
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)

        # print(img,' ',label)
        # print(img.shape)
        return img, label
    def __len__(self):
        return len(self.imgs)

# 輸入資料
train_data=MyDataset(place='./nine/train')
test_data=MyDataset(place='./nine/test')
# all data
num_train = len(train_data)
num_test = len(test_data)
print(num_train)
print(num_test)
# choose val data
indices = list(range(num_train))
indices_test = list(range(num_test))
split_val = int(np.floor(val_size * num_train))
np.random.shuffle(indices)
np.random.shuffle(indices_test)
train_idx, val_idx = indices[split_val:], indices[:split_val]
test_indx = indices_test[:num_test]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetRandomSampler(test_indx)

num_train = len(train_sampler)
num_val = len(val_sampler)
num_test = len(test_sampler)
print('training data:',num_train)
print('val data:',num_val)
print('testing data:',num_test)
# 資料分批
train_loader = Data.DataLoader(dataset=train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=0)
val_loader = Data.DataLoader(dataset=train_data, sampler=val_sampler, batch_size=BATCH_SIZE, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, sampler=test_sampler,batch_size=BATCH_SIZE, num_workers=0)

# print('train ->',train_loader)
# print('val ->',val_loader)
# print('test ->',test_loader)

# 辨識1-9的神經網路
class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, stride=1, padding=4), # 16 * 480 * 480
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4), # 16 * 120 * 120
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 32 * 120 * 120
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=4, stride=4), # 32 * 30 * 30
        )
        self.layers3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 * 30 * 30
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),  # 64 * 30 * 30
        )
        self.layers4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128 * 10 * 10
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 * 5 * 5
        )
        self.layers5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256 * 5 * 5
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 256,1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Linear(256,9)
        )
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.layers5(x)
        x = x.view(x.size(0),-1)
        output = self.fc(x)

        return output
# 複製網路
cnn = NET().to(device)
# 顯示神經架構
print(cnn)
# 設定參數函式與loss函式
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()
# 計算運算時間開始
torch.cuda.synchronize()
start = time.time()
# 開始訓練
print('\nStart Training')
for epoch in range(EPOCH):
    # reset

    running_loss = 0.0
    train_acc = 0.0
    cnn.train(True)
    print('\nepoch {}'.format(epoch + 1))
    for step, (x, y) in enumerate(train_loader):

        print('itr:',step,'/',len(train_loader))
        # train focus
        b_x, b_y = Variable(x).to(device), Variable(y).to(device)
        optimizer.zero_grad()

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        loss.backward()
        optimizer.step()
        # 觀察loss有無發散,顯示訓練準確率

        running_loss += loss.item()
        pred = torch.max(output, 1)[1]
        train_correct = (pred == b_y).sum()
        train_acc += train_correct.item()
    print('Train Loss: {:.6f}, Accuracy: {:.6f}'.format(running_loss / num_train, train_acc / num_train))
    cnn.eval()
    with torch.no_grad():

        val_loss = 0.0
        val_acc = 0.0
        for step, (x, y) in enumerate(val_loader):
            # test focus
            b_x, b_y = Variable(x).to(device), Variable(y).to(device)
            val_output = cnn(b_x)
            loss = loss_func(val_output, b_y)
            # 觀察loss有無發散,顯示測試準確率
            val_loss += loss.item()
            pred = torch.max(val_output, 1)[1]
            val_correct = (pred == b_y).sum()
            val_acc += val_correct.item()
        print('Val Loss: {:.6f}, Accuracy: {:.6f}'.format(val_loss / num_val, val_acc / num_val))
print('Finished Trainin\n')

# 開始test
print('Start Testing')
cnn.eval()
with torch.no_grad():
    # data preprocessing
    test_loss = 0.0
    test_acc = 0.0
    actuals = []
    predicted = []
    class_correct = list(0. for i in range(classes_len))
    class_total = list(0. for i in range(classes_len))
    for step, (x, y) in enumerate(test_loader):
        print('itr:', step, '/', len(test_loader))
        # test focus
        b_x, b_y = Variable(x).to(device), Variable(y).to(device)
        test_output = cnn(b_x)
        loss = loss_func(test_output, b_y)
        # 觀察loss有無發散,顯示測試準確率
        test_loss += loss.item()
        pred = torch.max(test_output, 1)[1]
        test_correct = (pred == b_y).sum()
        test_acc += test_correct.item()

        # 彙整測試結果的圖片
        correct_num = (pred == b_y).squeeze()
        actuals.extend(b_y.view_as(pred))
        predicted.extend(pred)
        for i in range(10 // 4):  # 根據batch size來設計
            label = b_y[i]
            class_correct[label] += correct_num[i].item()  # 計算每一個class預測正確的數量
            class_total[label] += 1  # 計算該class共有幾張圖片
actuals_class = [i.item() for i in actuals] # for matrix
predictions_class = [i.item() for i in predicted] # for matrix


print('Test Loss: {:.6f}, Accuracy: {:.6f}'.format(test_loss / num_test, test_acc / num_test))
print('Finished Testing!!!\n')



# display actuals
print('Confusion matrix:')
print(confusion_matrix(actuals_class, predictions_class))
cm = confusion_matrix(actuals_class, predictions_class)
plt.figure(figsize=(classes_len,classes_len))
#計算運算時間結束
torch.cuda.synchronize()
end = time.time()
use_time = time.localtime(end - start)
print('\n\n運行時間：\n%2d 分 %2d 秒 ' % (use_time.tm_min, use_time.tm_sec))
# 顯示矩陣
df_cm = pd.DataFrame(cm, index = [i for i in classes],columns = [i for i in classes])
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()


