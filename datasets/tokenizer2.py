#
# Original code: https://www.kaggle.com/yasufuminakama/inchi-resnet-lstm-with-attention-starter
#

class Tokenizer(object):

    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions


def load_tokenizer(data_root='/data/bms/'):
    import pickle
    from pathlib import Path
    saved_dicts = pickle.load(open(Path(data_root) / 'tokenizer2.pickle', 'rb'))
    tokenizer = Tokenizer()
    tokenizer.stoi = saved_dicts['stoi']
    tokenizer.itos = saved_dicts['itos']
    return tokenizer


if __name__ == '__main__':
    t = load_tokenizer('/data/bms/')
    print(f"tokenizer.stoi: {t.stoi}")
    print(f"tokenizer.stoi: {t.itos}")
