import argparse
import os
import shutil
    
from utils import set_seed, load_config, greedy_decode
from dataset import load_data
from encoder import EncoderCRNN
from decoder.model.transformer import Transformer
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(cfg_file):
    cfg = load_config(cfg_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    log_dir = cfg["generic"].get("log_dir")
    if cfg['generic'].get('clear_log') and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    set_seed(seed=cfg["generic"].get("seed"))

    train_loader, val_loader, test_loader, note2idx, idx2note, vocab_size, SOS_IDX, EOS_IDX, PAD_IDX = load_data(data_cfg=cfg["data"])

    encoder = EncoderCRNN(img_channels=1, layers=cfg["encoder"].get("layers"), no_lstm_layers=cfg["encoder"].get("no_lstm_layers")).to(device)
    decoder = Transformer(vocab_size=vocab_size, 
                          d_model=cfg["decoder"].get("d_model"), 
                          n_head=cfg["decoder"].get("n_head"), 
                          max_len=cfg["decoder"].get("max_len"),
                          ffn_hidden=cfg["decoder"].get("ffn_hidden"), 
                          n_layers=cfg["decoder"].get("n_layers"), 
                          drop_prob=cfg["decoder"].get("drop_prob"), 
                          device=device).to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=float(cfg["encoder"].get("lr")))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=float(cfg["decoder"].get("lr")))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    epochs = cfg["training"].get("epochs")

    # Save stuff
    save = cfg["training"].get("save")
    save_every = cfg["training"].get("save_every")
    save_dir = cfg["training"].get("save_dir")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    decode = cfg["training"].get("decode")
    decode_every = cfg["training"].get("decode_every")

    for epoch in range(epochs):
        print(epoch)

        encoder.train()
        decoder.train()
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            image, tgt, tgt_pad_mask = batch
            image, tgt, tgt_pad_mask = image.to(device), tgt.to(device), tgt_pad_mask.to(device)

            tgt_input = tgt[:, :-1] # Remove last EOS
            tgt_pad_mask = tgt_pad_mask[:, :-1]

            tgt_output = tgt[:, 1:]  # Remove first SOS
            
            loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            im_emb = encoder(image)
            output = decoder(im_emb, tgt_input, tgt_pad_mask)

            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            # Tensorboard
            global_step = epoch*len(train_loader) + batch_idx
            writer.add_scalar('Loss/train', loss, global_step)

        # Decode example
        if decode and epoch % decode_every == 0:
            encoder.eval()
            decoder.eval()
            tokens = greedy_decode(encoder, decoder, test_loader, PAD_IDX, SOS_IDX, EOS_IDX, device)
            writer.add_text('Decoded/train', str(tokens), global_step)

        # Val
        val_loss = 0
        encoder.eval()
        decoder.eval()
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            image, tgt, tgt_pad_mask = batch
            image, tgt, tgt_pad_mask = image.to(device), tgt.to(device), tgt_pad_mask.to(device)

            tgt_input = tgt[:, :-1] # Remove last EOS
            tgt_pad_mask = tgt_pad_mask[:, :-1]
            
            tgt_output = tgt[:, 1:]  # Remove first SOS
            
            with torch.no_grad():
                im_emb = encoder(image)
                output = decoder(im_emb, tgt_input, tgt_pad_mask)

                val_loss += criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))

        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, global_step)

        # Save model
        if save and epoch % save_every == 0:
            save_loc = os.path.join(save_dir, "model_epoch_{:}.pt".format(epoch))
            torch.save(model.state_dict(), save_loc)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AHAHAAH")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(cfg_file=args.config)
