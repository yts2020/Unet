import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import Unet
from data_load import MyDataset
from torch.utils.data import DataLoader


def train(model, train_loader, optimizer, epoch):
    model.train()
    all_step = 0
    for i in range(epoch):
        for step, data in enumerate(train_loader):
            all_step = all_step + 1
            input, target = data[0].to(torch.device('cuda')), data[1].to(torch.device('cuda'))
            output = model(input)
            loss = nn.BCELoss()(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0:
                print('epoch:', i + 1, 'step:', step + 1, 'loss:', loss)
    torch.save(model.state_dict(), './model/unet.pth')


def inference(model, model_path, img_dir):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    img_list = os.listdir(img_dir)
    with torch.no_grad():
        for img_name in img_list:
            img = cv2.imread(os.path.join(img_dir, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
            img = np.array(img / 255, dtype=np.float32)
            img = torch.from_numpy(img)
            img = img.to(torch.device('cuda'))
            output = model(img)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            output = output.cpu().numpy()
            output = np.squeeze(output, 0)
            output = np.transpose(output, (1, 2, 0))
            output = np.array(output * 255, dtype='uint8')
            cv2.imwrite('./result/' + img_name, output)


if __name__ == '__main__':
    unet = Unet().to(torch.device('cuda'))
    optimizer = optim.Adam(unet.parameters(), lr=0.00001)
    train_data = MyDataset(base_path='./data')
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    train(unet, train_loader, optimizer, 100)
    # model_path = './model/unet.pth'
    # img_dir = './test_img'
    # inference(unet, model_path, img_dir)
