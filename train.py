import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import translate, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

from seq2seq_attention import Encoder, Attention, Decoder, Seq2Seq
from loader import get_loader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    # Loading...
    train_loader, dataset = get_loader(
        root_folder="./DATA/train_feat",
        annotation_file="./DATA/train_data.csv"
    )

    # Training hyperparameters
    num_epochs = 1
    learning_rate = 3e-4
    batch_size = 32

    # Model hyperparameters
    input_size_encoder = 512
    input_size_decoder = len(dataset.vocab)
    output_size = len(dataset.vocab)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    enc_dropout = 0.2
    dec_dropout = 0.2

    # Tensorboard to get nice loss plot
    writer = SummaryWriter(f"runs/loss_plot")
    step = 0

    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)

    attention = Attention(hidden_size)

    decoder_net = Decoder(
        attention,
        input_size_decoder,
        decoder_embedding_size,
        hidden_size,
        output_size,
        num_layers,
        dec_dropout,
    ).to(device)

    model = Seq2Seq(encoder_net, decoder_net, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = dataset.vocab.stoi["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        for batch_idx, (vid, captions) in enumerate(train_loader):
            # Get input and targets and get to cuda
            inp_data = vid.to(device)
            target = captions.to(device)
            # target: (trg_len, batch_size)

            # Forward prop
            output = model(inp_data, target, len(dataset.vocab))
            # Output: (trg_len, batch_size, output_dim)

            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            # Back prop
            loss.backward()

            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # Plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

    test_loader, _ = get_loader(
        root_folder="./DATA/test_feat",
        annotation_file="./DATA/test_data.csv",
        batch_size=1
    )

    score = bleu(test_loader, model, dataset, device)
    print(f"Bleu score {score * 100:.2f}")

if __name__ ==  '__main__':
    train()