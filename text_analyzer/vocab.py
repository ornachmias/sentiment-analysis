import numpy as np
from collections import Counter
from gensim.parsing import preprocess_string


class Vocab:
    def __init__(self, texts):
        self.vocab = self._get_vocabulary(texts)

    @staticmethod
    def _get_vocabulary(texts):
        sentences = [preprocess_string(text) for text in texts]
        count_words = Counter(set(word for lst in sentences for word in lst))
        total_words = len(count_words)
        sorted_words = count_words.most_common(total_words)
        vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}
        return vocab_to_int

    @staticmethod
    def _pad_features(encoded_texts, seq_length):
        features = np.zeros((len(encoded_texts), seq_length), dtype=int)

        for i, review in enumerate(encoded_texts):
            review_len = len(review)
            if review_len <= seq_length:
                zeroes = list(np.zeros(seq_length - review_len))
                new = zeroes + review
            else:
                new = review[0:seq_length]

            features[i, :] = np.array(new)

        return features

    def encode(self, texts, seq_length):
        processed_texts = [preprocess_string(text) for text in texts]
        encoded_texts = []
        for text in processed_texts:
            encoded_texts.append([self.vocab[w] for w in text])
        return self._pad_features(encoded_texts, seq_length)
