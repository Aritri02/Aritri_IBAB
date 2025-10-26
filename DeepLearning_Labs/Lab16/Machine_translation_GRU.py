import os
import math
import random
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sacrebleu
from torch.cuda.amp import autocast, GradScaler

# ============================================================
#               CONFIGURATION AND HYPERPARAMETERS
# ============================================================
CSV_PATH = "Hindi_English_Truncated_Corpus.csv"  # parallel Hindi-English dataset
SP_PREFIX = "spm"  # prefix for SentencePiece model files
SRC_LANG = "src"  # generic source column name
TGT_LANG = "tgt"  # generic target column name
VOCAB_SIZE = 200  # small vocab for demo (increase for better results)
MAX_LEN = 128
BATCH_SIZE = 64
EMB_DIM = 256  # embedding dimension
HID_DIM = 512  # GRU hidden dimension
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
N_EPOCHS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER_FORCING_RATIO = 0.5  # probability to use target token as next input during training
PATIENCE = 5  # early stopping
SAVE_DIR = "./saved_gru_models"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
#                  CSV LOADING FUNCTION
# ============================================================
def load_parallel_csv(path):
    """
    Load a parallel corpus CSV file containing English–Hindi pairs.
    Auto-detect source/target columns by name (fallback: first two columns).
    """
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    src_candidates = ['source', 'src', 'english', 'en', 'sent_en']
    tgt_candidates = ['target', 'tgt', 'hindi', 'hi', 'sent_hi']

    # Try to detect which columns are English and Hindi
    src_col = next((df.columns[i] for i, c in enumerate(cols) if c in src_candidates), None)
    tgt_col = next((df.columns[i] for i, c in enumerate(cols) if c in tgt_candidates), None)

    # Default: first two columns if detection fails
    if src_col is None or tgt_col is None:
        src_col, tgt_col = df.columns[0], df.columns[1]
        print(f"[WARN] Using first two columns as parallel sentences: {src_col}, {tgt_col}")

    df = df[[src_col, tgt_col]].dropna()
    df.columns = [SRC_LANG, TGT_LANG]
    return df


# ============================================================
#            SENTENCEPIECE TOKENIZER TRAINING / LOADING
# ============================================================
def train_sentencepiece_from_series(series, out_prefix, vocab_size=200, model_type='unigram'):
    """
    Train a SentencePiece model on a pandas Series of text data.
    This converts text into subword token sequences.
    """
    tmp_txt = f"{out_prefix}.tmp.txt"
    series.to_csv(tmp_txt, index=False, header=False)

    spm.SentencePieceTrainer.Train(
        input=tmp_txt,
        model_prefix=out_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type=model_type,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        hard_vocab_limit=False
    )
    os.remove(tmp_txt)

    # Load the trained model
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{out_prefix}.model")
    return sp


# ============================================================
#                 DATASET AND COLLATE FUNCTION
# ============================================================
class ParallelDataset(Dataset):
    """
    Custom dataset for bilingual sentence pairs.
    Handles tokenization, truncation, and BOS/EOS addition.
    """

    def __init__(self, df, src_sp, tgt_sp, max_len=128):
        self.src_texts = df[SRC_LANG].astype(str).tolist()
        self.tgt_texts = df[TGT_LANG].astype(str).tolist()
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def encode_src(self, text):
        ids = [self.src_sp.bos_id()] + self.src_sp.EncodeAsIds(text)[:self.max_len - 2] + [self.src_sp.eos_id()]
        return torch.tensor(ids, dtype=torch.long)

    def encode_tgt(self, text):
        ids = [self.tgt_sp.bos_id()] + self.tgt_sp.EncodeAsIds(text)[:self.max_len - 2] + [self.tgt_sp.eos_id()]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        return self.encode_src(self.src_texts[idx]), self.encode_tgt(self.tgt_texts[idx])


def collate_fn(batch):
    """
    Collate function for DataLoader: pad variable-length sequences in a batch.
    Returns padded src/tgt tensors and masks.
    """
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    src_mask = (src_padded != 0)
    tgt_mask = (tgt_padded != 0)
    return src_padded, tgt_padded, src_mask, tgt_mask


# ============================================================
#                   ENCODER: GRU
# ============================================================
class EncoderGRU(nn.Module):
    """
    Encoder GRU: takes source tokens → context vector (final hidden state).
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.hid_dim = hid_dim

    def forward(self, src, src_mask=None):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, emb_dim)
        outputs, hidden = self.gru(embedded)  # hidden: (n_layers, batch, hid_dim)
        return outputs, hidden


# ============================================================
#                   DECODER: GRU
# ============================================================
class DecoderGRU(nn.Module):
    """
    Decoder GRU: generates target tokens one at a time.
    No attention — uses last hidden state as context.
    """

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token, hidden):
        """
        One decoding step.
        input_token: (batch,)
        hidden: (n_layers, batch, hid_dim)
        """
        input_token = input_token.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, emb_dim)
        output, hidden = self.gru(embedded, hidden)  # (batch, 1, hid_dim)
        pred = self.fc_out(output.squeeze(1))  # (batch, vocab)
        return pred, hidden


# ============================================================
#                   SEQ2SEQ WRAPPER
# ============================================================
class Seq2SeqGRU(nn.Module):
    """
    Combines Encoder and Decoder.
    Performs teacher forcing during training.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hid_dim == decoder.gru.hidden_size
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Full forward pass: encode source → decode target.
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=self.device)

        _, hidden = self.encoder(src)  # get encoder hidden state
        input_tok = tgt[:, 0]  # first decoder input = <bos>

        for t in range(1, tgt_len):
            preds, hidden = self.decoder.forward_step(input_tok, hidden)
            outputs[:, t, :] = preds
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = preds.argmax(1)
            input_tok = tgt[:, t] if teacher_force else top1  # next input

        return outputs

    def translate_greedy(self, src, src_sp, tgt_sp, max_len=80):
        """
        Greedy decoding for inference.
        """
        self.eval()
        with torch.no_grad():
            _, hidden = self.encoder(src)
            input_tok = torch.tensor([tgt_sp.bos_id()] * src.size(0),
                                     dtype=torch.long, device=self.device)
            outputs = []
            for _ in range(max_len):
                preds, hidden = self.decoder.forward_step(input_tok, hidden)
                top1 = preds.argmax(1)
                outputs.append(top1.unsqueeze(1))
                input_tok = top1
            outputs = torch.cat(outputs, dim=1)

        # Convert token IDs to readable text
        results = []
        for i in range(outputs.size(0)):
            ids = outputs[i].tolist()
            tokens = []
            for id_ in ids:
                if id_ == tgt_sp.eos_id():
                    break
                tokens.append(id_)
            text = tgt_sp.DecodeIds(
                [x for x in tokens if x not in (tgt_sp.pad_id(), tgt_sp.bos_id())]
            )
            results.append(text)
        return results


# ============================================================
#              TRAINING AND EVALUATION FUNCTIONS
# ============================================================
def train_epoch(model, dataloader, optimizer, criterion, scaler, teacher_forcing_ratio):
    """
    Train model for one epoch with mixed precision (AMP).
    """
    model.train()
    total_loss = 0
    for src, tgt, src_mask, tgt_mask in dataloader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            output = model(src, tgt, teacher_forcing_ratio)
            output_dim = output.shape[-1]
            # ignore <bos> position for loss
            output = output[:, 1:, :].reshape(-1, output_dim)
            tgt_y = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt_y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, src_sp, tgt_sp):
    """
    Evaluate model on validation set (no teacher forcing).
    """
    model.eval()
    total_loss = 0
    preds, refs = [], []
    with torch.no_grad():
        for src, tgt, src_mask, tgt_mask in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            tgt_y = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt_y)
            total_loss += loss.item()

            # Decode translations for BLEU
            batch_preds = model.translate_greedy(src, src_sp, tgt_sp, max_len=MAX_LEN)
            for i in range(len(batch_preds)):
                preds.append(batch_preds[i])
                ref_ids = [int(x) for x in tgt[i].tolist()
                           if x not in (tgt_sp.bos_id(), tgt_sp.eos_id(), tgt_sp.pad_id())]
                refs.append(tgt_sp.DecodeIds(ref_ids))
    bleu = sacrebleu.corpus_bleu(preds, [refs]) if len(preds) > 0 else None
    return total_loss / len(dataloader), bleu.score if bleu else None


# ============================================================
#                       MAIN TRAINING PIPELINE
# ============================================================
def main():
    print("[INFO] Loading CSV...")
    df = load_parallel_csv(CSV_PATH)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_frac = 0.05 if len(df) > 1000 else 0.1
    n_val = int(len(df) * val_frac)
    df_train, df_val = df[:-n_val], df[-n_val:]
    print(f"[INFO] Train size: {len(df_train)}, Val size: {len(df_val)}")

    # Train or load SentencePiece tokenizers
    src_sp_path = f"{SP_PREFIX}.{SRC_LANG}.model"
    tgt_sp_path = f"{SP_PREFIX}.{TGT_LANG}.model"
    if not (os.path.exists(src_sp_path) and os.path.exists(tgt_sp_path)):
        print("[INFO] Training SentencePiece models...")
        src_sp = train_sentencepiece_from_series(df_train[SRC_LANG], f"{SP_PREFIX}.{SRC_LANG}", VOCAB_SIZE)
        tgt_sp = train_sentencepiece_from_series(df_train[TGT_LANG], f"{SP_PREFIX}.{TGT_LANG}", VOCAB_SIZE)
    else:
        src_sp = spm.SentencePieceProcessor();
        src_sp.Load(src_sp_path)
        tgt_sp = spm.SentencePieceProcessor();
        tgt_sp.Load(tgt_sp_path)
        print("[INFO] Loaded SentencePiece models.")

    src_vocab_size, tgt_vocab_size = src_sp.GetPieceSize(), tgt_sp.GetPieceSize()
    print(f"[INFO] src vocab: {src_vocab_size}, tgt vocab: {tgt_vocab_size}")

    # Create datasets & dataloaders
    train_dataset = ParallelDataset(df_train, src_sp, tgt_sp, MAX_LEN)
    val_dataset = ParallelDataset(df_val, src_sp, tgt_sp, MAX_LEN)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize encoder-decoder model
    encoder = EncoderGRU(src_vocab_size, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT, pad_idx=0)
    decoder = DecoderGRU(tgt_vocab_size, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT, pad_idx=0)
    model = Seq2SeqGRU(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler('cuda')
    best_bleu, no_improve = -1e9, 0

    # ---------------------------
    #        TRAIN LOOP
    # ---------------------------
    for epoch in range(1, N_EPOCHS + 1):
        print(f"=== Epoch {epoch}/{N_EPOCHS} ===")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, TEACHER_FORCING_RATIO)
        val_loss, val_bleu = evaluate(model, val_loader, criterion, src_sp, tgt_sp)
        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}  Val BLEU={val_bleu if val_bleu else 'N/A'}")

        # Save best BLEU model
        if val_bleu and val_bleu > best_bleu:
            best_bleu = val_bleu
            no_improve = 0
            ckpt_path = os.path.join(SAVE_DIR, f"best_gru_epoch{epoch}_bleu{val_bleu:.2f}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'src_sp_model': src_sp_path,
                'tgt_sp_model': tgt_sp_path,
                'bleu': val_bleu
            }, ckpt_path)
            print(f"[INFO] Saved new best model to {ckpt_path}")
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

    print("[INFO] Training complete.")

    # ---------------------------
    #    EXAMPLE TRANSLATIONS
    # ---------------------------
    model.eval()
    sample_src, sample_tgt = next(iter(val_loader))
    sample_src = sample_src.to(DEVICE)[:8]
    translations = model.translate_greedy(sample_src, src_sp, tgt_sp, MAX_LEN)
    print("\n=== Example translations (greedy) ===")
    for i, trans in enumerate(translations):
        print(f"{i + 1}. {trans}")


if __name__ == "__main__":
    main()
