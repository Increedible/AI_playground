import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import re
from collections import Counter
import time

# Define functions and classes
def generate_jsons(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                text += f.read() + " "
    text = text.replace("‘", "'").replace("’", "'").replace('“', '"').replace('”','"')
    def tokenize(text):
        text = re.sub(r'\s+', ' ', text)  # Replace all whitespace characters with a single space
        tokens = list(text)  # Character-level tokenization
        return tokens

    tokens = tokenize(text)
    counter = Counter(tokens)
    sorted_vocab = sorted(counter, key=counter.get, reverse=True)
    vocab_size = len(sorted_vocab)
    word_to_idx = {word: idx for idx, word in enumerate(sorted_vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(sorted_vocab)}

    with open('Samples/word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f)
    with open('Samples/idx_to_word.json', 'w') as f:
        json.dump(idx_to_word, f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    generate_jsons('Samples/Books/')
    if torch.cuda.is_available():
        print(f"[AGENT] Is CUDA supported by this system? {torch.cuda.is_available()}")
        print(f"[AGENT] CUDA version: {torch.version.cuda}")
        cuda_id = torch.cuda.current_device()
        print(f"[AGENT] ID of current CUDA device: {torch.cuda.current_device()}")
        print(f"[AGENT] Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
    else:
        print("[AGENT] Cuda is not available; training will be done on CPU.")

with open('Samples/word_to_idx.json', 'r') as f:
    word_to_idx = json.load(f)
with open('Samples/idx_to_word.json', 'r') as f:
    idx_to_word = json.load(f)

vocab_size = len(word_to_idx)

text = ""
for filename in os.listdir('Samples/Books/'):
    if filename.endswith(".txt"):
        with open(os.path.join('Samples/Books/', filename), 'r', encoding='utf-8') as f:
            text += f.read() + " "

print("Replacing quote symbols... ", end="")
time.sleep(0.1)
text = text.replace("‘", "'").replace("’", "'").replace('“', '"').replace('”','"')
time.sleep(0.1)
print("Done.")

def tokenize(text):
    text = re.sub(r'\s+', ' ', text)  # Replace all whitespace characters with a single space
    tokens = list(text)  # Character-level tokenization
    return tokens

print("Tokenising... ", end="")
time.sleep(0.1)
tokens = tokenize(text)
time.sleep(0.1)
print("Done.")
print("Preparing Model... ", end="")
time.sleep(0.1)

class BibleDataset(Dataset):
    def __init__(self, tokens, word_to_idx, seq_length=100):
        self.tokens = tokens
        self.word_to_idx = word_to_idx
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_length]
        y = self.tokens[idx + 1:idx + self.seq_length + 1]
        x = torch.tensor([self.word_to_idx[word] for word in x], dtype=torch.long)
        y = torch.tensor([self.word_to_idx[word] for word in y], dtype=torch.long)
        return x, y

dataset = BibleDataset(tokens, word_to_idx)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(num_layers, batch_size, hidden_dim).zero_().to(device),
                  weight.new(num_layers, batch_size, hidden_dim).zero_().to(device))
        return hidden

embedding_dim = 256  # Reduced embedding dimension
hidden_dim = 256     # Reduced hidden dimension
num_layers = 3       # Reduced number of layers

model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
torch.backends.cudnn.benchmark = True

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

time.sleep(0.1)
print("Done.")

start_epoch = 0
checkpoint_path = 'Samples/checkpoint.pth'
if os.path.exists(checkpoint_path):
    print("Loading in previous model... ", end="")
    time.sleep(0.1)
    start_epoch, _ = load_checkpoint(checkpoint_path)
    time.sleep(0.1)
    print("Done.")

num_epochs = 10
model.train()

if __name__ == '__main__':
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        ami = len(dataloader)
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model.init_hidden(batch_size=x.size(0))
            hidden = tuple([each.data for each in hidden])
            output, hidden = model(x, hidden)
            loss = criterion(output, y.view(-1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Step: {i}/{ami}, Loss: {loss.item()} DONT QUIT...', end="")
                save_checkpoint(model, optimizer, epoch, total_loss / len(dataloader), checkpoint_path)
                print(" Saved... Free to quit")

        scheduler.step(total_loss / len(dataloader))
        print(f'Epoch: {epoch+1}, Learning Rate: {scheduler.get_last_lr()}')

@torch.no_grad()
def generate_text(model, start_text, length=250):
    model.eval()
    generated = start_text
    words = list(start_text)  # Character-level tokenization for input text
    hidden = model.init_hidden(batch_size=1)

    for _ in range(length):
        x = torch.tensor([[word_to_idx[word] for word in words]], dtype=torch.long).to(device)
        output, hidden = model(x, hidden)
        word_weights = output[-1].squeeze().div(0.8).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        word = idx_to_word[str(word_idx.item())]
        generated += word
        words.append(word)
        words = words[1:]

    return generated

if __name__ == '__main__':
    while True:
        start_text = input("Prompt: ")
        print(generate_text(model, start_text))
