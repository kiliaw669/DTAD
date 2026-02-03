import torch
import os
import torch.nn as nn
from utils import *
import libs.autoencoder

def trainer(model, category, config):
    '''
    Training the UNet model
    :param model: the UNet model
    :param category: the category of the dataset
    '''
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay
    )
    train_dataset = AD_Dataset(
        root= config.data.data_dir,
        category=category,
        config = config,
        is_train=True,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_path = '/oss-mt-sysu-release/wangzh/data/autoencoder/autoencoder_kl.pth'
    autoencoder = libs.autoencoder.get_model(pretrained_path=pretrained_path)
    autoencoder.to(device)

    for epoch in range(config.model.epochs):
        for step, batch in enumerate(trainloader):
            optimizer.zero_grad()
            
            _z = autoencoder.encode(batch[0].to(device))
            print(_z.shape)

            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()
            loss = get_loss(model, _z, t, config) 

            print("loss:", loss.item())
            raise
            
            t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()
            loss = get_loss(model, batch[0], t, config) 
            loss.backward()
            optimizer.step()
            if (epoch+1) % 25 == 0 and step == 0:
                print(f"Epoch {epoch+1} | Loss: {loss.item()}")
            if (epoch+1) %250 == 0 and epoch>0 and step ==0:
                if config.model.save_model:
                    model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch+1)))
                
