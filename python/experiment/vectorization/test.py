import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors, Phrases
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from python.experiment.sentiment.experiment5a import MyExperiment


class SentimentCNN(nn.Module):
    """
    The embedding layer + CNN model that will be used to perform sentiment analysis.
    """

    def __init__(self, embed_model, vocab_size, output_size, embedding_dim,
                 num_filters=100, kernel_sizes=[3, 4, 5], freeze_embeddings=True, drop_prob=0.5):
        super(SentimentCNN, self).__init__()
        """
        Initialize the model by setting up the layers.
        """

        # set class vars
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim

        # 1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # set weights to pre-trained
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors))  # all vectors
        # (optional) freeze embedding weights
        if freeze_embeddings:
            self.embedding.requires_grad = False

        # 2. convolutional layers
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=(k - 2, 0))
            for k in kernel_sizes])

        # 3. final, fully-connected layer for classification
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size)

        # 4. dropout and sigmoid layers
        self.dropout = nn.Dropout(drop_prob)
        self.sig = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        """
        Convolutional + max pooling layer
        """
        # squeeze last dim to get size: (batch_size, num_filters, conv_seq_length)
        # conv_seq_length will be ~ 200
        x = F.relu(conv(x)).squeeze(3)

        # 1D pool over conv_seq_length
        # squeeze to get size: (batch_size, num_filters)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x):
        """
        Defines how a batch of inputs, x, passes through the model layers.
        Returns a single, sigmoid-activated class score as output.
        """
        # embedded vectors
        embeds = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        # embeds.unsqueeze(1) creates a channel dimension that conv layers expect
        embeds = embeds.unsqueeze(1)

        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]

        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)

        # final logit
        logit = self.fc(x)

        # sigmoid-activated --> a class score
        return self.sig(logit)


w2v = KeyedVectors.load('../../data/word2vec.model')
vocab_size = len(w2v.vocab)
output_size = 3
embedding_dim = 100
num_filters = 100
kernel_sizes = [3, 4, 5]
net = SentimentCNN(w2v, vocab_size, output_size, embedding_dim, num_filters, kernel_sizes)

grain = 'CORN'
print(grain)
experiment = MyExperiment(grain)
df = experiment.combined_df


def tokenize(embed_lookup, words):
    tokenized = []
    for word in words:
        try:
            idx = embed_lookup.vocab[word].index
        except:
            idx = 0
        tokenized.append(idx)
    return tokenized


df['tokenized'] = df['new content'].swifter.apply(lambda c: tokenize(w2v, c))
df['direction'] = df['direction'].swifter.apply(lambda d: d + 1)
df['direction'] = df['direction'].shift(-1)
df = df[df['direction'].notnull()]


# df['direction'] = np.array(df['direction'], dtype=int)


def pad_then_truncate(tokenized, seq_length):
    return tokenized[:seq_length] if len(tokenized) >= seq_length else np.pad(tokenized, pad_width=(
        seq_length - len(tokenized), 0), constant_values=(0,))[:seq_length]


# designated_seq_length = int(np.round(np.median(list(map(lambda arr: len(arr), df['tokenized'])))))
designated_seq_length = 300

df['tokenized'] = df['tokenized'].swifter.apply(lambda arr: pad_then_truncate(arr, designated_seq_length))

x_train, x_remaining, y_train, y_remaining = train_test_split(df['tokenized'], df['direction'], test_size=0.2,
                                                              random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_remaining, y_remaining, test_size=0.5, random_state=42)

loader_train = DataLoader(
    TensorDataset(torch.from_numpy(np.array(list(x_train))), torch.from_numpy(np.array(list(y_train)))),
    shuffle=True,
    batch_size=50
)
loader_valid = DataLoader(
    TensorDataset(torch.from_numpy(np.array(list(x_valid))), torch.from_numpy(np.array(list(y_valid)))),
    shuffle=True,
    batch_size=50
)
loader_test = DataLoader(
    TensorDataset(torch.from_numpy(np.array(list(x_test))), torch.from_numpy(np.array(list(y_test)))),
    shuffle=True,
    batch_size=50
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# training loop


def train(net, train_loader, epochs, print_every=100):
    counter = 0  # for printing

    # train for some number of epochs
    net.train()
    for e in range(epochs):

        # batch loop
        for inputs1, labels1 in train_loader:
            counter += 1

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output1 = net(inputs1)

            # calculate the loss and perform backprop

            loss = criterion(output1.squeeze(), labels1.long())
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_losses = []
                net.eval()
                for i, l in loader_valid:
                    o = net(i)
                    val_loss = criterion(o.squeeze(), l.long())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))


epochs = 10  # this is approx where I noticed the validation loss stop decreasing
print_every = 100

train(net, loader_train, epochs, print_every=print_every)

# Get test data loss and accuracy

test_losses = []  # track loss
num_correct = 0

net.eval()
# iterate over test data
for inputs, labels in loader_test:
    # get predicted outputs
    output = net(inputs)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.long())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.argmax(output, dim=1)  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.long().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)

# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(loader_test.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
