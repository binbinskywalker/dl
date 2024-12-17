import string

class Vectorizer:
    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)
    
    def tokenize(self, text):
        text = self.standardize(text)
        return text.split()
    
    def make_vocabulary(self, dataset):
        self.vocabulary = {"": 0, "[UNK]":1}
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                print("token", token)
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
                    print("self.vocabulary", self.vocabulary)
        self.inverse_vocabulary = dict((v, k) for v, k in self.vocabulary.items())
        print("self.inverse_vocabulary", self.inverse_vocabulary)

    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        result = ""
        for i in int_sequence:
            for k,v in self.inverse_vocabulary.items():
                if v == i:
                    result = result + " "+ k
        return result

    # def decode(self, int_sequence):
    #     return " ".join(
    #         self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence)
