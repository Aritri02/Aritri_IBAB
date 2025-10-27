#!/usr/bin/env python3
import os
from pathlib import Path
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter
import nltk

# ---------------------------
# Config / Hyperparameters
# ---------------------------
data_location = "/home/ibab/PythonProject/DL/Lab15/archive"
images_path = os.path.join(data_location, "Images")
captions_file = os.path.join(data_location, "captions.txt")  # CSV with (image_filename, caption)

embed_size = 256
hidden_size = 512
num_epochs = 5
batch_size = 32
learning_rate = 1e-3
max_len = 20
freq_threshold = 5
num_layers = 1
seed = 42

# Reproducibility
random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Ensure NLTK tokenizer is available
# ---------------------------
# punkt is required by word_tokenize
nltk.download("punkt", quiet=True)

# ---------------------------
# Load captions CSV
# ---------------------------
# Expecting a CSV with at least two columns: image filename and caption.
# If your captions file has headers use header=0, else header=None
try:
    df = pd.read_csv(captions_file, header=None)
except Exception as e:
    print(f"Error reading {captions_file}: {e}")
    raise

# If dataframe has header row with names like 'image,caption', you may prefer:
# df = pd.read_csv(captions_file)   # and then use df['image'], df['caption']

# ---------------------------
# Vocabulary helper
# ---------------------------
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        # reserved tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return nltk.word_tokenize(str(text).lower())

    def build_vocab(self, sentence_list):
        freqs = Counter()
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            freqs.update(tokens)

        for word, freq in freqs.items():
            if freq >= self.freq_threshold:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx

    def numericalize(self, text):
        tokenized = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized
        ]

# Build vocabulary from captions column (assume captions in column 1)
captions_col = df.iloc[:, 1].astype(str).tolist()
vocab = Vocabulary(freq_threshold=freq_threshold)
vocab.build_vocab(captions_col)
print("Vocab size:", len(vocab))

# ---------------------------
# Transforms & Dataset
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class FlickrDataset(Dataset):
    def __init__(self, dataframe, img_dir, vocab, transform=None, max_len=20):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        caption = str(self.df.iloc[idx, 1])

        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Numerical caption with tokens [<SOS>, ...words..., <EOS>]
        tokens = [self.vocab.stoi["<SOS>"]]
        tokens += self.vocab.numericalize(caption)
        tokens.append(self.vocab.stoi["<EOS>"])

        # Pad / truncate to max_len
        if len(tokens) < self.max_len:
            tokens += [self.vocab.stoi["<PAD>"]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return image, torch.tensor(tokens, dtype=torch.long)

# Read raw (try no-header first, we'll detect)
raw_df = pd.read_csv(captions_file, header=None, dtype=str)

# If the first row contains header-like values, re-read with header=0
first_cell = str(raw_df.iloc[0, 0]).strip().lower()
if any(h in first_cell for h in ("image", "filename", "file", "photo")):
    print("Detected header row in CSV -> re-reading with header=0")
    df = pd.read_csv(captions_file, header=0, dtype=str)
else:
    df = raw_df.copy()

# Ensure at least two columns exist
if df.shape[1] < 2:
    raise ValueError(f"Expected >=2 columns in captions file but found {df.shape[1]}")

# Identify image and caption columns (first two columns)
img_col = df.columns[0]
cap_col = df.columns[1]

# Clean whitespace
df[img_col] = df[img_col].astype(str).str.strip()
df[cap_col] = df[cap_col].astype(str).str.strip()

# Optionally drop rows where img_col looks like header text (safer)
mask_header_like = df[img_col].str.lower().isin(["image", "filename", "image_name", "photo"])
if mask_header_like.any():
    print(f"Dropping {mask_header_like.sum()} rows that look like header values in column {img_col}")
    df = df[~mask_header_like].reset_index(drop=True)

# Resolve missing extensions: if filename doesn't exist, try common extensions
images_path = images_path  # keep using your variable
exts = [".jpg", ".jpeg", ".png"]

def resolve_filename(name):
    name = str(name).strip()
    p = Path(images_path) / name
    if p.exists():
        return name
    # if name already has an extension but file doesn't exist, return None
    if Path(name).suffix:
        return None
    # try adding common extensions
    for e in exts:
        if (Path(images_path) / (name + e)).exists():
            return name + e
    return None

# Apply resolution and filter
df["resolved_name"] = df[img_col].apply(resolve_filename)
missing = df["resolved_name"].isnull()
print(f"Total rows: {len(df)}; missing images after resolution: {missing.sum()}")

# Print a few missing examples for debugging
if missing.any():
    print("Examples of missing filenames (first 10):", df.loc[missing, img_col].head(10).tolist())

# Keep only rows where images exist
df = df[~missing].copy()
df.reset_index(drop=True, inplace=True)
# replace image column with resolved_name
df[img_col] = df["resolved_name"]
df.drop(columns=["resolved_name"], inplace=True)

print("After filtering, rows kept:", len(df))
print("Sample rows:")
print(df[[img_col, cap_col]].head(10))

# Recreate dataset / dataloader with debug-friendly settings
dataset = FlickrDataset(df, images_path, vocab, transform, max_len=max_len)
# use num_workers=0 while debugging to see errors in main process
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# ---------------------------
# Feature extractor: ResNet18
# ---------------------------
# Load pretrained ResNet18 and drop final FC
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
modules = list(resnet.children())[:-1]  # remove final fc
resnet = nn.Sequential(*modules)
resnet.to(device)
resnet.eval()
for p in resnet.parameters():
    p.requires_grad = False

# ---------------------------
# LSTM-based Captioning Model
# ---------------------------
class ImageCaptionLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptionLSTM, self).__init__()
        self.fc_img = nn.Linear(512, embed_size)  # ResNet18 last conv output flattened -> 512
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, img_features, captions):
        """
        img_features: (B, 512)  -- raw ResNet features flattened
        captions: (B, L)        -- token ids (with <SOS> at start)
        """
        # Encode image and prepare as first timestep embedding
        img_emb = self.fc_img(img_features).unsqueeze(1)  # (B,1,embed_size)
        embeddings = self.embed(captions[:, :-1])         # (B,L-1,embed_size)

        # Concatenate image embedding at front -> sequence length L
        inputs = torch.cat((img_emb, embeddings), dim=1)  # (B, L, embed_size) where L == max_len

        outputs, (h_n, c_n) = self.lstm(inputs)           # outputs: (B, L, hidden)
        outputs = self.fc_out(outputs)                    # (B, L, vocab_size)
        return outputs

# Initialize model, loss, optimizer
model = ImageCaptionLSTM(embed_size, hidden_size, len(vocab), num_layers=num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------------------
# Training loop
# ---------------------------
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (imgs, caps) in enumerate(dataloader):
        imgs = imgs.to(device)
        caps = caps.to(device)  # (B, max_len)

        # Extract features from ResNet (no grad)
        with torch.no_grad():
            feats = resnet(imgs).view(imgs.size(0), -1)  # (B,512)

        # Forward
        outputs = model(feats, caps)  # (B, L, V)
        loss = criterion(outputs.reshape(-1, outputs.size(2)), caps.reshape(-1))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        # Optional: gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 50 == 0 or i == 0:
            avg = running_loss / (i + 1)
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, AvgLoss: {avg:.4f}")

    print(f"Epoch {epoch+1} completed. Avg loss: {running_loss/len(dataloader):.4f}")

print("Training finished.")

# ---------------------------
# Caption generation (inference)
# ---------------------------
def generate_caption(model, image_path, vocab, max_len=20):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = resnet(image).view(1, -1)              # (1,512)
        feat = model.fc_img(feat).unsqueeze(1)        # (1,1,embed_size)

        # initialize LSTM states
        h = torch.zeros(model.num_layers, 1, model.hidden_size).to(device)
        c = torch.zeros(model.num_layers, 1, model.hidden_size).to(device)

        generated = [vocab.stoi["<SOS>"]]
        for t in range(max_len):
            last_token = torch.tensor([generated[-1]], dtype=torch.long).to(device)
            emb = model.embed(last_token).unsqueeze(1)  # (1,1,embed_size)

            # feed image embedding for first step, otherwise feed last token embedding
            inputs = feat if len(generated) == 1 else emb  # (1,1,embed_size)
            out, (h, c) = model.lstm(inputs, (h, c))       # out: (1,1,hidden)
            out = model.fc_out(out.squeeze(1))             # (1, vocab_size)
            pred = out.argmax(dim=1).item()
            generated.append(pred)
            if pred == vocab.stoi["<EOS>"]:
                break

    words = [vocab.itos.get(idx, "<UNK>") for idx in generated]
    # Remove <SOS> and <EOS> from returned string
    # If EOS not found, drop first token (SOS) only
    if words and words[-1] == "<EOS>":
        words = words[1:-1]
    else:
        words = words[1:]
    return " ".join(words)

# ---------------------------
# Example usage
# ---------------------------
# pick an index (make sure dataframe has that many entries)
test_index = min(50, len(df) - 1)
test_img = os.path.join(images_path, str(df.iloc[test_index, 0]))
print("Example image path:", test_img)

try:
    caption = generate_caption(model, test_img, vocab, max_len=max_len)
    print("Generated caption:", caption)
except Exception as e:
    print("Error generating caption:", e)



