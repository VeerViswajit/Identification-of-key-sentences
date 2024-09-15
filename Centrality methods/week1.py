import networkx as nx
from collections import defaultdict
import string

class GraphProcessor:
    def __init__(self, sentences, window=10, tags=None, tag_delim='/'):
        self.graph = nx.Graph()
        self.sentences = sentences
        self.window = window
        self.tags = tags or ['JJ', 'NNP', 'NNS', 'NN']
        self.tag_delim = tag_delim
        self.build_graph()

    def preprocess_text(self, sentence):
        # Remove punctuation and split into words
        translator = str.maketrans('', '', string.punctuation)
        return [word.split(self.tag_delim)[0].translate(translator) for word in sentence.split()]

    def build_graph(self):
        for sentence in self.sentences:
            words = self.preprocess_text(sentence)
            for word in words:
                if word not in self.graph:
                    self.graph.add_node(word)
            for i, word1 in enumerate(words):
                for j in range(i + 1, min(len(words), i + self.window)):
                    word2 = words[j]
                    if word1 != word2:
                        if not self.graph.has_edge(word1, word2):
                            self.graph.add_edge(word1, word2, weight=0)
                        self.graph[word1][word2]['weight'] += 1

    def connected_components(self):
        components = list(nx.connected_components(self.graph))
        phrases = []
        for comp in components:
            if len(comp) > 1:  # Only consider meaningful keyphrases
                subgraph = self.graph.subgraph(comp)
                for clique in nx.find_cliques(subgraph):
                    phrase = " ".join(clique)       #cliques (fully connected subgraphs) within the component.
                    phrases.append(phrase)
        return phrases
    
    def degree_centrality_keyphrases(self):
        degree = nx.degree_centrality(self.graph)
        keyphrases = [phrase for phrase, deg in degree.items()]
        return keyphrases

    def evaluate(self, candidates, reference_set):
        if not candidates or not reference_set:
            print("Evaluation error: Candidates or reference set is empty.")
            return {}
        
        ranks = []
        average_precision = 0.0
        for i, candidate in enumerate(candidates):
            for ref in reference_set:
                if candidate in ref or ref in candidate:  # Flexible matching
                    ranks.append(i + 1)
                    average_precision += len(ranks) / float(i + 1)
                    break
        
        keys_at_5 = sum(1 for rank in ranks if rank <= 5)
        keys_at_10 = sum(1 for rank in ranks if rank <= 10)
        
        scores = {
            'P5': keys_at_5 / 5.0,
            'R5': keys_at_5 / float(len(reference_set)),
            'F5': (2.0 * keys_at_5 / 5.0 * keys_at_5 / float(len(reference_set))) / (keys_at_5 / 5.0 + keys_at_5 / float(len(reference_set))) if (keys_at_5 / 5.0 + keys_at_5 / float(len(reference_set))) > 0 else 0.0,
            'P10': keys_at_10 / 10.0,
            'R10': keys_at_10 / float(len(reference_set)),
            'F10': (2.0 * keys_at_10 / 10.0 * keys_at_10 / float(len(reference_set))) / (keys_at_10 / 10.0 + keys_at_10 / float(len(reference_set))) if (keys_at_10 / 10.0 + keys_at_10 / float(len(reference_set))) > 0 else 0.0,
            'R-MAX': len(ranks) / float(len(reference_set)),  #Recall at Maximum Rank
            'MAP': average_precision / len(reference_set)  #Mean Average Precision
        }
        return scores

    def print_graph(self):
        print("Graph nodes and edges with weights:")
        for node in self.graph.nodes:
            print(f"Node: {node}")
        for edge in self.graph.edges(data=True):
            print(f"Edge: {edge[0]} - {edge[1]}, Weight: {edge[2]['weight']}")

if __name__ == "__main__":
    # Define the input text
    text_input = '''modeling/vbg self-consistent/jj multi-class/jj dynamic/jj traffic/nn flow/nn ./punct
    in/in this/dt study/nn ,/punct we/prp present/vbp a/dt systematic/jj self-consistent/jj multiclass/jj multilane/nn traffic/nn model/nn derived/vbn from/in the/dt vehicular/jj boltzmann/jj equation/nn and/cc the/dt traffic/nn dispersion/nn model/nn ./punct
    the/dt multilane/jj domain/nn is/vbz considered/vbn as/in a/dt two-dimensional/jj space/nn and/cc the/dt interaction/nn among/in vehicles/nns in/in the/dt domain/nn is/vbz described/vbn by/in a/dt dispersion/nn model/nn ./punct
    the/dt reason/nn we/prp consider/vbp a/dt multilane/jj domain/nn as/in a/dt two-dimensional/jj space/nn is/vbz that/in the/dt driving/vbg behavior/nn of/in road/nn users/nns may/md not/rb be/vb restricted/vbn by/in lanes/nns ,/punct especially/rb motorcyclists/nns ./punct
    the/dt dispersion/nn model/nn ,/punct which/wdt is/vbz a/dt nonlinear/jj poisson/nnp equation/nn ,/punct is/vbz derived/vbn from/in the/dt car-following/jj theory/nn and/cc the/dt equilibrium/nn assumption/nn ./punct
    under/in the/dt concept/nn that/in all/dt kinds/nns of/in users/nns share/vbp the/dt finite/jj section/nn ,/punct the/dt density/nn is/vbz distributed/vbn on/in a/dt road/nn by/in the/dt dispersion/nn model/nn ./punct
    in/in addition/nn ,/punct the/dt dynamic/jj evolution/nn of/in the/dt traffic/nn flow/nn is/vbz determined/vbn by/in the/dt systematic/jj gas-kinetic/jj model/nn derived/vbn from/in the/dt boltzmann/jj equation/nn ./punct
    multiplying/vbg boltzmann/jj equation/nn by/in the/dt zeroth/nn ,/punct first-/jj and/cc second-order/jj moment/nn functions/nns ,/punct integrating/vbg both/dt side/nn of/in the/dt equation/nn and/cc using/vbg chain/nn rules/nns ,/punct we/prp can/md derive/vb continuity/nn ,/punct motion/nn and/cc variance/nn equation/nn ,/punct respectively/rb ./punct
    however/rb ,/punct the/dt second-order/jj moment/nn function/nn ,/punct which/wdt is/vbz the/dt square/nn of/in the/dt individual/jj velocity/nn ,/punct is/vbz employed/vbn by/in previous/jj researches/vbz does/vbz not/rb have/vb physical/jj meaning/nn in/in traffic/nn flow/nn ./punct'''

    # Define reference set for evaluation
    reference_set = {
        "self-consistent multiclass dynamic traffic flow modeling", 
        "multilane traffic model", 
        "vehicular Boltzmann equation", 
        "traffic dispersion model",
        "road users", 
        "nonlinear Poisson equation", 
        "car-following theory", 
        "dynamic evolution", 
        "variance equation", 
        "motion equation", 
        "Poisson equation"
    }

    # Split the input text into sentences
    sentences = text_input.split('\n')

    # # Create a GraphProcessor instance with the sentences
    graph_processor = GraphProcessor(sentences, window=10, tags=['JJ', 'NNP', 'NNS', 'NN'])

    # Extract meaningful keyphrases using connected components
    keyphrases_degree = graph_processor.connected_components()

    # # Filter the keyphrases to find those related to the reference set
    filtered_keyphrases = []
    for keyphrase in keyphrases_degree:
        for ref in reference_set:
            if any(word in ref for word in keyphrase.split()):
                filtered_keyphrases.append(keyphrase)
                break

    # # Print the graph
    # graph_processor.print_graph()

    # # Print the top 10 keyphrases related to the reference set
    top_keyphrases = filtered_keyphrases[:10]
    print("\nTop 10 Keyphrases related to the reference set:")
    for phrase in top_keyphrases:
        print(phrase)

    # # Evaluate the extracted keyphrases against the reference set
    if top_keyphrases:  # Check if keyphrases are not empty
        evaluation_scores = graph_processor.evaluate(top_keyphrases, reference_set)
    else:
        print("No meaningful keyphrases extracted.")
        evaluation_scores = {}

    # Print evaluation scores
    print("\nEvaluation Scores:")
    for metric, value in evaluation_scores.items():
        print(f"{metric}: {value:.4f}")

    text1 = "New Delhi is the capital of India, India has 35 states and Delhi is one of the state, the Delhi's population is 1/3rd of India's Population, Milk is white in colour"
    sentences1 = text1.split(',')
    graph_processor1 = GraphProcessor(sentences1, window=10, tags=['JJ', 'NNP', 'NNS', 'NN'])
    graph_processor1.print_graph()
# Top 10 Keyphrases related to the reference set:
# this  we study in
# poisson  is a nonlinear which
# poisson  is a nonlinear equation
# poisson  is derived equation
# variance  and motion continuity
# variance  and motion equation
# variance  and respectively equation
# continuity  derive motion can
# continuity  derive motion and
# restricted users be not may

# Evaluation Scores:
# P5: 0.0000
# R5: 0.0000
# F5: 0.0000
# P10: 0.1000
# R10: 0.0909
# F10: 0.0952
# R-MAX: 0.0909
# MAP: 0.0152

# Top 10 Keyphrases related to the reference set:
#  the functions equation and both integrating secondorder first moment zeroth
#  the functions equation and both integrating secondorder first moment side of
#  the functions equation and both integrating using side of
#  the functions equation and by secondorder moment first zeroth
#  the functions equation and by secondorder moment first of
#  the both integrating equation and rules chain side using of
#  the restricted be road users by not may of
#  the chain rules equation using and we side of
#  the chain rules equation using and we can derive
#  the chain rules equation using and we can of

# Evaluation Scores:
# P5: 0.0000
# R5: 0.0000
# F5: 0.0000
# P10: 0.1000
# R10: 0.0909
# F10: 0.0952
# R-MAX: 0.0909
# MAP: 0.0130