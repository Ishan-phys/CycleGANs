import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import myDataset
from utils import save_checkpoint, load_checkpoint
from model.discriminator import Discriminator
from model.generator import Generator
from train import train_func


def main():
    # Load the dataset from data folder
    train_dataset = myDataset(
        root_A=config.TRAIN_DIR+"/A_train", 
        root_B=config.TRAIN_DIR+"/B_train", 
        transform=config.transforms
    )

    loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    # Initialize the Generator and the Discriminator models
    disc_A = Discriminator(config.INPUT_SHAPE).to(config.DEVICE)
    disc_B = Discriminator(config.INPUT_SHAPE).to(config.DEVICE)
    gen_AB = Generator(config.INPUT_SHAPE, num_residual_blocks=config.RES_BLOCKS).to(config.DEVICE)
    gen_BA = Generator(config.INPUT_SHAPE, num_residual_blocks=config.RES_BLOCKS).to(config.DEVICE)

    # Set the optimizer and the loss function
    optimizer_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    optimizer_gen = optim.Adam(
        list(gen_AB.parameters()) + list(gen_BA.parameters()),
        lr = config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    # Initialize the training function
    training = train_func(disc_A, disc_B, gen_AB, gen_BA, optimizer_disc, optimizer_gen, L1, mse)
    
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A, gen_BA, optimizer_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_B, gen_AB, optimizer_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_A, disc_A, optimizer_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_B, disc_B, optimizer_disc, config.LEARNING_RATE,
        )
        
    for epoch in range(config.NUM_EPOCHS):
        training.train(loader)
        
if __name__ == "__main__":
    main()
    
    
