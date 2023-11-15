
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

"""Hyperparameters"""
"""Number of independent sequences we can process in parallel"""
batch_size = 32
"""Maximum context length for predictions"""
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
"""Use a GPU if you have one available"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embeds = 32

"""
Read Shakespeare file
Command: wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""
with open('input.txt', 'r', encoding='utf-8') as file:
	text = file.read()

unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)

"""Tokenization"""
"""Map characters to integers"""
string_to_int = {char: idx for idx, char in enumerate(unique_chars)}
int_to_string = {idx: char for idx, char in enumerate(unique_chars)}

"""Takes a string as input and outputs a list of integers"""
encode = lambda string: [string_to_int[char] for char in string]
"""Takes a list of integers as input and outputs string"""
decode = lambda integers: ''.join([int_to_string[idx] for idx in integers])

"""Data Storage"""
"""Encode entire text and store in a toch tensor"""
data = torch.tensor(encode(text), dtype=torch.long)

"""Split data into training (90%) and validation (10%)"""
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(split:str):
	"""The function generates a small batch of data: inputs x and targets y"""
	data = train_data if split == 'train' else val_data

	"""Generate 4 random numbers between 0 and n - block_size
	These will be the starting indices of the 4 sequences in the batch"""
	n = len(data)
	rand_starting_points = torch.randint(n - block_size, (batch_size,))
	"""
	Stack up the inputs and targets for each sequence in the batch into a 4 x 8 tensor
	The target sequence for each input sequence is the same as the input sequence, but shifted by one character to the right
	"""
	"""Converted x and y to tuples in order to avoid an error in the next cell"""
	x = torch.stack(tuple(data[point:point + block_size] for point in rand_starting_points))
	y = torch.stack(tuple(data[point + 1:point + block_size + 1] for point in rand_starting_points))
	"""When device becomes cuda, send x and y to it"""
	x, y = x.to(device), y.to(device)
	return x, y

"""Tells PyTorch that we don't intend to do backprop here"""
@torch.no_grad()
def estimate_loss():
	out = {}
	"""
	Set model to evaluation phase
	Although there's no real difference between these phases for this model, it's good practice to do this
	"""
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for iter in range(eval_iters):
			X, Y = get_batch(split)
			logits, loss = model(X, Y)
			"""Store loss on each iteration"""
			losses[iter] = loss.item()
		"""Store mean loss for each split"""
		out[split] = losses.mean()
	"""Set model to training phase"""
	model.train()
	return out

"""Simplified Bigram Model"""
class BigramLanguageModel(nn.Module):

	def __init__(self):
		super().__init__()
		"""Instead of directly reads the logits for the next token using a lookup table of embeddings, go through an intermediate step first"""
		self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

	def forward(self, idx, targets=None):
		"""idx and targets are both integer tensors with dimensions (Batch, Time); here (4, 8)"""
		"""logits is a tensor with dimensions (Batch, Time, Channel); here (4, 8, 65)"""
		logits = self.token_embedding_table(idx)

		if targets is None:
			loss = None
		else:
			"""
			Need to reshape logits because the cross entropy loss function expects a tensor with 2 dimensions: (Batch * Channel, Time)
			Need to reshape targets because the cross entropy loss function expects a tensor with one dimension: (Batch * Time)
			"""
			batch, time, channel = logits.shape
			logits = logits.view(batch * time, channel)
			targets = targets.view(batch * time)
			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def generate(self, idx, max_new_tokens):
		"""Generates new tokens given a context of existing tokens such that an array of (batch, time) indices becomes (batch, time + 1))"""
		"""idx is (batch, time) array of indices in the current context"""
		for _ in range (max_new_tokens):
			"""
			Get predictions
			Calling self this way invokes the forward method and supplies idx as an argument
			"""
			logits, loss = self(idx)
			"""Focus only on the most recent time step and transform logits into an array of (Batch, Channel)"""
			logits = logits[:, -1, :]
			"""Apply softmax to get probabilities for each token in the vocabulary"""
			probs = F.softmax(logits, dim=-1)
			"""
			Sample once from the probability distribution to get the next token
			Array shape: (Batch, 1)
			"""
			idx_next = torch.multinomial(probs, num_samples=1)
			"""Append sample index to the running sequence so the final array is (Batch, Time + 1)"""
			idx = torch.cat((idx, idx_next), dim=1)
		return idx

model = BigramLanguageModel()
device_model = model.to(device)

"""Train Bigram Model"""
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):
	"""Every once in a while, evaluate the loss on train and val sets"""
	if iter % eval_interval == 0:
		losses = estimate_loss()
		"""May need to change this to output"""
		print(f"Iteration {iter}: Training Loss: {losses['train']:.4f}; Validation Loss: {losses['val']:.4f}")

	"""Sample data"""
	xbatch, ybatch = get_batch('train')
	"""Forward pass and evaluate loss"""
	logits, loss = model(xbatch, ybatch)
	optimizer.zero_grad(set_to_none=True)
	"""Backward pass to compute gradients"""
	loss.backward()
	"""Use gradients to update parameters"""
	optimizer.step()

"""Generate new text"""
context = torch.zeros((1, 1), dtype=torch.long, device=device)
result = decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist())
print('Bigram Language Model Generated Text:')
print(result)
