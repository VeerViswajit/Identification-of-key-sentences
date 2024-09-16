import re
import networkx as nx
import numpy as np
from nltk.stem import SnowballStemmer
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine
# import torch

class SemanticGraphProcessor:
    def __init__(self, sentences, window=10, tags=None, tag_delim='/'):
        self.graph = nx.Graph()
        self.sentences = sentences
        self.window = window
        self.tags = tags or ['JJ', 'NNP', 'NNS', 'NN']
        self.tag_delim = tag_delim
        self.keyphrase_candidates = set()
        self.stemmer = SnowballStemmer("english")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.build_graph()

    def get_word_embedding(self, word):
        inputs = self.tokenizer(word, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

    def build_graph(self):
        embeddings = {}
        for sentence in self.sentences:
            words = sentence.split()
            filtered_words = [
                self.stemmer.stem(word.split(self.tag_delim)[0])
                for word in words
                if word.split(self.tag_delim)[0].lower() not in {'the','a','as','is','by','in','and','of','we','not','from','which','that'} and not re.match(r'\W', word)
            ]
            for word in filtered_words:
                if word not in embeddings:
                    embeddings[word] = self.get_word_embedding(word)
                if word not in self.graph:
                    self.graph.add_node(word)
            for i, word1 in enumerate(filtered_words):
                for j in range(i + 1, min(len(filtered_words), i + self.window)):
                    word2 = filtered_words[j]
                    if word1 != word2:
                        sim = 1 - cosine(embeddings[word1], embeddings[word2])
                        if not self.graph.has_edge(word1, word2):
                            self.graph.add_edge(word1, word2, weight=sim)
                        else:
                            self.graph[word1][word2]['weight'] += sim

    def textrank_keyphrases(self, top_n=10):
        pagerank_scores = nx.pagerank(self.graph, weight='weight')
        sorted_keyphrases = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        keyphrases = [
            phrase for phrase, _ in sorted_keyphrases[:top_n]
            if phrase not in {'the','a','as'} and not re.match(r'\W', phrase)
        ]
        return keyphrases

    def evaluate(self, candidates, reference_set):
        ranks = []
        average_precision = 0.0
        for i, candidate in enumerate(candidates):
            if candidate in reference_set:
                ranks.append(i + 1)
                average_precision += len(ranks) / float(i + 1)
        keys_at_5 = sum(1 for rank in ranks if rank <= 5)
        keys_at_10 = sum(1 for rank in ranks if rank <= 10)
        scores = {
            'P5': keys_at_5 / 5.0,
            'R5': keys_at_5 / float(len(reference_set)),
            'F5': 2.0 * (keys_at_5 / 5.0 * (keys_at_5 / float(len(reference_set)))) / (keys_at_5 / 5.0 + keys_at_5 / float(len(reference_set))) if (keys_at_5 / 5.0 + keys_at_5 / float(len(reference_set))) > 0 else 0.0,
            'P10': keys_at_10 / 10.0,
            'R10': keys_at_10 / float(len(reference_set)),
            'F10': 2.0 * (keys_at_10 / 10.0 * (keys_at_10 / float(len(reference_set)))) / (keys_at_10 / 10.0 + keys_at_10 / float(len(reference_set))) if (keys_at_10 / 10.0 + keys_at_10 / float(len(reference_set))) > 0 else 0.0,
            'R-MAX': len(ranks) / float(len(reference_set)),
            'MAP': average_precision / len(reference_set)
        }
        return scores

if __name__ == "__main__":
    text_input = '''modeling/vbg self-consistent/jj multi-class/jj dynamic/jj traffic/nn flow/nn ./punct
    in/in this/dt study/nn ,/punct we/prp present/vbp a/dt systematic/jj self-consistent/jj multiclass/jj multilane/nn traffic/nn model/nn derived/vbn from/in the/dt vehicular/jj boltzmann/jj equation/nn and/cc the/dt traffic/nn dispersion/nn model/nn ./punct
    the/dt multilane/jj domain/nn is/vbz considered/vbn as/in a/dt two-dimensional/jj space/nn and/cc the/dt interaction/nn among/in vehicles/nns in/in the/dt domain/nn is/vbz described/vbn by/in a/dt dispersion/nn model/nn ./punct
    the/dt reason/nn we/prp consider/vbp a/dt multilane/jj domain/nn as/in a/dt two-dimensional/jj space/nn is/vbz that/in the/dt driving/vbg behavior/nn of/in road/nn users/nns may/md not/rb be/vb restricted/vbn by/in lanes/nns ,/punct especially/rb motorcyclists/nns ./punct
    the/dt dispersion/nn model/nn ,/punct which/wdt is/vbz a/dt nonlinear/jj poisson/nnp equation/nn ,/punct is/vbz derived/vbn from/in the/dt car-following/jj theory/nn and/cc the/dt equilibrium/nn assumption/nn ./punct
    under/in the/dt concept/nn that/in all/dt kinds/nns of/in users/nns share/vbp the/dt finite/jj section/nn ,/punct the/dt density/nn is/vbz distributed/vbn on/in a/dt road/nn by/in the/dt dispersion/nn model/nn ./punct
    in/in addition/nn ,/punct the/dt dynamic/jj evolution/nn of/in the/dt traffic/nn flow/nn is/vbz determined/vbn by/in the/dt systematic/jj gas-kinetic/jj model/nn derived/vbn from/in the/dt boltzmann/jj equation/nn ./punct
    multiplying/vbg boltzmann/jj equation/nn by/in the/dt zeroth/nn ,/punct first-/jj and/cc second-order/jj moment/nn functions/nns ,/punct integrating/vbg both/dt side/nn of/in the/dt equation/nn and/cc using/vbg chain/nn rules/nns ,/punct we/prp can/md derive/vb continuity/nn ,/punct motion/nn and/cc variance/nn equation/nn ,/punct respectively/rb ./punct
    however/rb ,/punct the/dt second-order/jj moment/nn function/nn ,/punct which/wdt is/vbz the/dt square/nn of/in the/dt individual/jj velocity/nn ,/punct is/vbz employed/vbn by/in previous/jj researches/vbz does/vbz not/rb have/vb physical/jj meaning/nn in/in traffic/nn flow/nn ./punct'''

    sentences = text_input.split('\n')

    # Create a SemanticGraphProcessor instance with the sentences
    graph_processor = SemanticGraphProcessor(sentences, window=10, tags=['JJ', 'NNP', 'NNS', 'NN'])

    # Get keyphrases using TextRank
    keyphrases_textrank = graph_processor.textrank_keyphrases()

    # Example reference set for evaluation
    reference_set = {"self-consistent multiclass dynamic traffic flow modeling", "multilane traffic model", "vehicular Boltzmann equation", "traffic dispersion model","road users", "nonlinear Poisson equation", "car-following theory", "dynamic evolution", "variance equation", "motion equation", "Poisson equation"}

    # Evaluate the keyphrases against the reference set
    evaluation_scores = graph_processor.evaluate(keyphrases_textrank, reference_set)

    # Print keyphrases
    print("Keyphrases using TextRank:")
    for phrase in keyphrases_textrank:
        print(phrase)

    # Print evaluation scores
    print("\nEvaluation Scores:")
    for metric, value in evaluation_scores.items():
        print(f"{metric}: {value:.4f}")
