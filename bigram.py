
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

"""Download Shakespeare training dataset"""
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

"""Read Shakespeare file"""
with open('input.txt', 'r', encoding='utf-8') as file:
	text = file.read()

print(f'The Shakespeare file has {len(text):,} characters')

"""Print first 1K characters"""
print(text[:1000])

unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)
print(f'The Shakespeare file has {vocab_size} unique characters')
str_chars = ''.join(unique_chars)
print(f'Vocabulary: {str_chars}')

"""Tokenization"""

"""Map characters to integers"""
string_to_int = {char: idx for idx, char in enumerate(unique_chars)}
int_to_string = {idx: char for idx, char in enumerate(unique_chars)}

"""Takes a string as input and outputs a list of integers"""
encode = lambda string: [string_to_int[char] for char in string]
"""Takes a list of integers as input and outputs string"""
decode = lambda integers: ''.join([int_to_string[idx] for idx in integers])


print(encode('test'))
print(decode(encode('test')))

"""Data Storage"""

"""Encode entire text and store in a toch tensor"""
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

print(data[:100])

"""Split data into training (90%) and validation (10%)"""
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

"""
Chunking
- Split training data into sampled chunks with a block size (token length) of 8
- For a block size of 8, we actually need 9 chars because the sample is the next predicted char, and we need some context for the next prediction. So, to predict 8 times, we need 9 chars
"""

block_size = 8
train_data[:block_size + 1]


"""
Here, X are the input characters and y are the target characters
Again, we need to offset by 1 for the targets so that the model has some context to predict from
"""
x = train_data[:block_size]
y = train_data[1:block_size + 1]


for t in range(block_size):
	"""Context is always the characters in x up to and including t"""
	context = x[:t + 1]
	target = y[t]
	print(f'When input is {context}, target is {target}')

"""
Batching and Blocking
- Batch size is the number of independent sequences we want to process in parallel on every forward-backward pass of the transformer
- Block size is the maximum content length in tokens used to make predictions
"""

batch_size = 4
block_size = 8

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

xbatch, ybatch = get_batch('train')

print('Inputs:')
print(xbatch.shape)
print(xbatch)

print('Targets:')
print(ybatch.shape)
print(ybatch)

# The tensor area (4 x 8) is the number of examples contained in the array. The above consists of 32 independent examples, from the transformer's perspective.

for batch in range(batch_size):
	print(f'Chunk {batch + 1}')
	for time in range(block_size):
		context = xbatch[batch, :time + 1]
		target = ybatch[batch, time]
		print(f'When input is {context.tolist()}, target is {target}')

# ## Baseline: Bigram Language Model
# Predicts probability of the next token in a sequence given the previous token. It is called a "bigram" model because it considers pairs of adjacent words in the sequence. The model is trained on a corpus of text and learns the probability distribution of words in the corpus. The model can then be used to generate new text by sampling from the learned distribution.


class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):
		super().__init__()
		"""Each token reads the logits for the next token using a lookup table of embeddings"""
		self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

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

"""
Why this architecture is silly: <br>
A bigram means predicting one character from another. We're only using the last character to predict the next one, but we're still feeding the model the entire context on each iteration
"""

model = BigramLanguageModel(vocab_size)
device_model = model.to(device)

"""Output is the logits for each of the 4 x 8 positions and the loss"""
logits, loss = model(xbatch, ybatch)
print(f'Shape of logits tensor: {logits.shape}')
print(f'Loss: {loss:.3f}')

"""
Since we're using a mathematical negative log loss function, we can estimate the ideal loss using our vocabulary size. <br>
Ideally, we want a loss of `-ln(1/65) = ~4.17`
"""

"""
Supply a tensor of zeros as the initial idx context
In vocab, zero represents a newline character, so it makes sense to start here
Predict 100 tokens, pull out the first sequence, which is a 1D array of all indices, and convert to a list
"""
result = decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist())
print(result)

"""Train Bigram Model"""

"""We can get away with a high learning rate since we're using a small network"""
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):
	"""Sample data"""
	xbatch, ybatch = get_batch('train')

	"""Forward pass and evaluate loss"""
	logits, loss = model(xbatch, ybatch)
	optimizer.zero_grad(set_to_none=True)
	"""Backward pass to compute gradients"""
	loss.backward()
	"""Use gradients to update parameters"""
	optimizer.step()

print(f'Loss after 10000 iterations: {loss.item():.3f}')

"""Generate new text"""
context = torch.zeros((1, 1), dtype=torch.long, device=device)
result = decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=400)[0].tolist())
print(result)

"""New result is still nonsense but also a dramatic improvement from the untrained model|"""
