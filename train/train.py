import argparse
import os
import shutil
import time
    
from utils import set_seed, load_config, greedy_decode, build_pad_mask, build_causal_mask, calc_acc
from dataset import load_data
from encoder import EncoderCRNN, Decoder
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
    run_log_dir = os.path.join(log_dir, cfg.get("name"), str(time.time()))
    train_writer = SummaryWriter(log_dir=os.path.join(run_log_dir, "train"))
    eval_writer = SummaryWriter(log_dir=os.path.join(run_log_dir, "eval"))

    set_seed(seed=cfg["generic"].get("seed"))

    train_loader, val_loader, test_loader, note2idx, idx2note, vocab_size, SOS_IDX, EOS_IDX, PAD_IDX = load_data(data_cfg=cfg["data"], seed=cfg["generic"].get("seed"))

    encoder = EncoderCRNN(img_channels=1, 
                          layers=cfg["encoder"].get("layers"), 
                          no_lstm_layers=cfg["encoder"].get("no_lstm_layers"),
                          lstm_hidden=cfg["decoder"].get("d_model")).to(device)
    """decoder = Transformer(vocab_size=vocab_size, 
                          d_model=cfg["decoder"].get("d_model"), 
                          n_head=cfg["decoder"].get("n_head"), 
                          max_len=cfg["data"].get("max_seq_len"),
                          ffn_hidden=cfg["decoder"].get("ffn_hidden"), 
                          n_layers=cfg["decoder"].get("n_layers"), 
                          drop_prob=cfg["decoder"].get("drop_prob"), 
                          device=device).to(device)"""
    decoder = Decoder(vocab_size=vocab_size, 
                          d_model=cfg["decoder"].get("d_model"), 
                          n_head=cfg["decoder"].get("n_head"), 
                          max_len=cfg["data"].get("max_seq_len"),
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
            image, tgt = batch
            image, tgt = image.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1] # Remove last EOS
            tgt_output = tgt[:, 1:]  # Remove first SOS

            tgt_pad_mask = build_pad_mask(tgt_input, PAD_IDX).to(device)
            tgt_causal_mask = build_causal_mask(tgt_input, cfg["decoder"].get("n_head"))
            
            loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            im_emb = encoder(image)
            output = decoder(im_emb, tgt_input, tgt_pad_mask, tgt_causal_mask)

            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            # Tensorboard
            global_step = epoch*len(train_loader) + batch_idx
            train_writer.add_scalar('loss', loss, global_step)
            train_writer.add_scalar('accuracy', calc_acc(output, tgt_output), global_step)

        # Decode example
        if decode and epoch % decode_every == 0:
            encoder.eval()
            decoder.eval()
            tgt, tokens = greedy_decode(encoder, decoder, test_loader, PAD_IDX, SOS_IDX, EOS_IDX, device)

            tgt_decoded = [idx2note[i] for i in tgt]
            tokens_decoded = [idx2note[i] for i in tokens]
            eval_writer.add_text('decoded', str(tokens_decoded), global_step)
            eval_writer.add_text('decoded target', str(tgt_decoded), global_step)

        # Val
        val_loss = 0
        val_acc = []
        encoder.eval()
        decoder.eval()
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            image, tgt = batch
            image, tgt = image.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1] # Remove last EOS
            tgt_output = tgt[:, 1:]  # Remove first SOS

            tgt_pad_mask = build_pad_mask(tgt_input, PAD_IDX).to(device)
            tgt_causal_mask = build_causal_mask(tgt_input, cfg["decoder"].get("n_head"))
            
            with torch.no_grad():
                im_emb = encoder(image)
                output = decoder(im_emb, tgt_input, tgt_pad_mask, tgt_causal_mask)

                val_loss += criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                val_acc.append(calc_acc(output, tgt_output))

        val_loss /= len(val_loader)
        eval_writer.add_scalar('loss', val_loss, global_step)
        eval_writer.add_scalar('accuracy', sum(val_acc) / len(val_acc), global_step)

        # Save model
        if save and epoch % save_every == 0:
            encoder_save_loc = os.path.join(save_dir, "encoder_epoch_{:}.pt".format(epoch))
            decoder_save_loc = os.path.join(save_dir, "decoder_epoch_{:}.pt".format(epoch))
            torch.save(encoder.state_dict(), encoder_save_loc)
            torch.save(decoder.state_dict(), decoder_save_loc)

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
