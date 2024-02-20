import gradio as gr
from transformers import pipeline
import re
import pickle
import torch
import torch.nn as nn
from torchtext.transforms import PadTransform
from torch.nn import functional as F
from tqdm import tqdm
from underthesea import text_normalize

# Build Vocabulary
device = "cpu"

# Build Vocabulary
MAX_LENGTH = 20
class Vocabulary:
    """The Vocabulary class is used to record words, which are used to convert
    text to numbers and vice versa.
    """

    def __init__(self, lang="vi"):
        self.lang = lang
        self.word2id = dict()
        self.word2id["<sos>"] = 0  # Start of Sentence Token
        self.word2id["<eos>"] = 1  # End of Sentence Token
        self.word2id["<unk>"] = 2  # Unknown Token
        self.word2id["<pad>"] = 3  # Pad Token
        self.sos_id = self.word2id["<sos>"]
        self.eos_id = self.word2id["<eos>"]
        self.unk_id = self.word2id["<unk>"]
        self.pad_id = self.word2id["<pad>"]
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.pad_transform = PadTransform(max_length = MAX_LENGTH, pad_value = self.pad_id)

    def __getitem__(self, word):
        """Return ID of word if existed else return ID unknown token
        @param word (str)
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """Return True if word in Vocabulary else return False
        @param word (str)
        """
        return word in self.word2id

    def __len__(self):
        """
        Return number of tokens(include sos, eos, unk and pad tokens) in Vocabulary
        """
        return len(self.word2id)

    def lookup_tokens(self, word_indexes: list):
        """Return the list of words by lookup by ID
        @param word_indexes (list(int))
        @return words (list(str))
        """
        return [self.id2word[word_index] for word_index in word_indexes]

    def add(self, word):
        """Add word to vocabulary
        @param word (str)
        @return index (str): index of the word just added
        """
        if word not in self:
            word_index = self.word2id[word] = len(self.word2id)
            self.id2word[word_index] = word
            return word_index
        else:
            return self[word]

    def preprocessing_sent(self, sent, lang="en"):
        """Preprocessing a sentence (depend on language english or vietnamese)
        @param sent (str)
        @param lang (str)
        """

        # Lowercase sentence and remove space at beginning and ending
        sent = sent.lower().strip()

        # Replace HTML charecterist
        sent = re.sub("&apos;", "'", sent)
        sent = re.sub("&quot;", '"', sent)
        sent = re.sub("&#91;", "[", sent)
        sent = re.sub("&#93;", "]", sent)

        # Remove unnecessary space
        sent = re.sub("(?<=\w)\.", " .", sent)

        # Normalizing the distance between tokens (word and punctuation)
        sent = re.sub("(?<=\w),", " ,", sent)
        sent = re.sub("(?<=\w)\?", " ?", sent)
        sent = re.sub("(?<=\w)\!", " !", sent)
        sent = re.sub(" +", " ", sent)

        if (lang == "en") or (lang == "eng") or (lang == "english"):
            # Replace short form
            sent = re.sub("what's", "what is", sent)
            sent = re.sub("who's", "who is", sent)
            sent = re.sub("which's", "which is", sent)
            sent = re.sub("who's", "who is", sent)
            sent = re.sub("here's", "here is", sent)
            sent = re.sub("there's", "there is", sent)
            sent = re.sub("it's", "it is", sent)

            sent = re.sub("i'm", "i am", sent)
            sent = re.sub("'re ", " are ", sent)
            sent = re.sub("'ve ", " have ", sent)
            sent = re.sub("'ll ", " will ", sent)
            sent = re.sub("'d ", " would ", sent)

            sent = re.sub("aren't", "are not", sent)
            sent = re.sub("isn't", "is not", sent)
            sent = re.sub("don't", "do not", sent)
            sent = re.sub("doesn't", "does not", sent)
            sent = re.sub("wasn't", "was not", sent)
            sent = re.sub("weren't", "were not", sent)
            sent = re.sub("won't", "will not", sent)
            sent = re.sub("can't", "can not", sent)
            sent = re.sub("let's", "let us", sent)

        else:
            # Package underthesea.text_normalize support to normalize vietnamese
            sent = text_normalize(sent)
        if not sent.endswith(('.', '!', '?')):
            sent = sent + ' .'
        return sent.strip()

    def tokenize_corpus(self, corpus, disable=False):
        """Split the documents of the corpus into words
        @param corpus (list(str)): list of documents
        @param disable (bool): notified or not
        @return tokenized_corpus (list(list(str))): list of words
        """
        if not disable:
            print("Tokenize the corpus...")
        tokenized_corpus = list()
        for document in tqdm(corpus, disable=disable):
            tokenized_document = ["<sos>"] + self.preprocessing_sent(document, self.lang).split(" ") + ["<eos>"]
            tokenized_corpus.append(tokenized_document)
        return tokenized_corpus

    def corpus_to_tensor(self, corpus, is_tokenized=False, disable=False):
        """Convert corpus to a list of indices tensor
        @param corpus (list(str) if is_tokenized==False else list(list(str)))
        @param is_tokenized (bool)
        @return indicies_corpus (list(tensor))
        """
        if is_tokenized:
            tokenized_corpus = corpus
        else:
            tokenized_corpus = self.tokenize_corpus(corpus, disable=disable)
        indicies_corpus = list()
        for document in tqdm(tokenized_corpus, disable=disable):
            indicies_document = torch.tensor(
                list(map(lambda word: self[word], document)), dtype=torch.int64
            )

            indicies_corpus.append(self.pad_transform(indicies_document))

        return indicies_corpus

    def tensor_to_corpus(self, tensor, disable=False):
        """Convert list of indices tensor to a list of tokenized documents
        @param indicies_corpus (list(tensor))
        @return corpus (list(list(str)))
        """
        corpus = list()
        for indicies in tqdm(tensor, disable=disable):
            document = list(map(lambda index: self.id2word[index.item()], indicies))
            corpus.append(document)

        return corpus


with open("vocab_source_final.pkl", "rb") as file:
    VOCAB_SOURCE = pickle.load(file)
with open("vocab_target_final.pkl", "rb") as file:
    VOCAB_TARGET = pickle.load(file)

input_embedding = torch.zeros((len(VOCAB_SOURCE), 100))
output_embedding = torch.zeros((len(VOCAB_TARGET), 100))


def create_input_emb_layer(pretrained = False):
    if not pretrained:
        weights_matrix = torch.zeros((len(VOCAB_SOURCE), 100))
    else:
        weights_matrix = input_embedding
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight.data = weights_matrix
    emb_layer.weight.requires_grad = False

    return emb_layer, embedding_dim

def create_output_emb_layer(pretrained = False):
    if not pretrained:
        weights_matrix = torch.zeros((len(VOCAB_TARGET), 100))
    else:
        weights_matrix = output_embedding
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight.data = weights_matrix
    emb_layer.weight.requires_grad = False

    return emb_layer, embedding_dim


class EncoderAtt(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout = 0.1):
        """ Encoder RNN
        @param input_dim (int): size of vocab_souce
        @param hidden_dim (int)
        @param dropout (float): dropout ratio of layer drop out
        """
        super(EncoderAtt, self).__init__()
        self.hidden_dim = hidden_dim
        # Using pretrained Embedding
        self.embedding, self.embedding_dim = create_input_emb_layer(True)
        self.gru = nn.GRU(self.embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        """ Bahdanau Attention
        @param hidden_size (int)
        """
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class DecoderAtt(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1):
        """ Decoder RNN using Attention
        @param hidden_size (int)
        @param output_size (int): size of vocab_target
        @param dropout (float): dropout ratio of layer drop out
        """
        super(DecoderAtt, self).__init__()
        # Using pretrained Embedding
        self.embedding, self.embedding_dim = create_output_emb_layer(True)
        self.fc = nn.Linear(self.embedding_dim, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(0)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            # Teacher forcing
            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.fc(self.embedding(input)))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


# Load VietAI Translation
envit5_translater = pipeline("translation", model="VietAI/envit5-translation")

INPUT_DIM = len(VOCAB_SOURCE)
OUTPUT_DIM = len(VOCAB_TARGET)
HID_DIM = 512

# Load our Model Translation
ENCODER = EncoderAtt(INPUT_DIM, HID_DIM)
ENCODER.load_state_dict(torch.load("encoderatt_epoch_35.pt", map_location=torch.device('cpu')))
DECODER = DecoderAtt(HID_DIM, OUTPUT_DIM)
DECODER.load_state_dict(torch.load("decoderatt_epoch_35.pt", map_location=torch.device('cpu')))


def evaluate_final_model(sentence, encoder, decoder, vocab_source, vocab_target, disable = False):
    """ Evaluation Model
    @param encoder (EncoderAtt)
    @param decoder (DecoderAtt)
    @param sentence (str)
    @param vocab_source (Vocabulary)
    @param vocab_target (Vocabulary)
    @param disable (bool)
    """
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = vocab_source.corpus_to_tensor([sentence], disable = disable)[0].view(1,-1).to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == vocab_target.eos_id:
                decoded_words.append('<eos>')
                break
            decoded_words.append(vocab_target.id2word[idx.item()])
    return decoded_words, decoder_attn

def translate_sentence(sentence):
    output_words, _ = evaluate_final_model(sentence, ENCODER, DECODER, VOCAB_SOURCE, VOCAB_TARGET, disable= True)
    if "<pad>" in output_words:
      output_words.remove("<pad>")
    if "<unk>" in output_words:
      output_words.remove("<unk>")
    if "<sos>" in output_words:
      output_words.remove("<sos>")
    if "<eos>" in output_words:
      output_words.remove("<eos>")

    return ' '.join(output_words).capitalize()


def envit5_translation(text):
    res = envit5_translater(
        text,
        max_length=512,
        early_stopping=True,
    )[0]["translation_text"][3:]
    return res


def translation(text):
    output1 = translate_sentence(text)

    if not text.endswith(('.', '!', '?')):
        text = text + '.'
    output2 = envit5_translation(text)

    return (output1, output2)

if __name__ == "__main__":
    examples = [["Hello guys", "Input"], 
                ["Xin chào các bạn", "Output"]]

    demo = gr.Interface(
        theme = gr.themes.Base(),
        fn=translation,
        title="Co Gai Mo Duong",
        description="""
        ## Machine Translation: English to Vietnamese
        """,
        examples=examples,
        inputs=[
            gr.Textbox(
                lines=5, placeholder="Enter text", label="Input"
            )
        ],
        outputs=[
            gr.Textbox(
                "text", label="Our Machine Translation"
            ),
            gr.Textbox(
                "text", label="VietAI Machine Translation"
            )
        ]
    )

    demo.launch(share = True)
