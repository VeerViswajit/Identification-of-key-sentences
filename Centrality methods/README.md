# Graph-Based Key Phrase Extraction Using Centrality Measures

This code implements a **graph-based keyphrase extraction** technique, where words are treated as nodes, and edges represent co-occurrence within a sliding window of 10 words. We implemented two centrality measures: **Degree Centrality** and **TextRank**, based on the findings from the referenced paper.

The methodology is designed to identify key phrases by analyzing the structure of the graph. The results are evaluated against a reference set using precision, recall, and F1 score.

**Referred from**:  
BOUDIN, F. (2013). A COMPARISON OF CENTRALITY MEASURES FOR GRAPH-BASED KEYPHRASE EXTRACTION. INTERNATIONAL JOINT CONFERENCE ON NATURAL LANGUAGE PROCESSING, 834–838.  
[Link to repository](https://github.com/boudinfl/centrality_measures_ijcnlp13)

### Paper Overview

The paper evaluated various centrality measures for keyphrase extraction. While several were compared, we implemented the best two: **Degree Centrality** and **TextRank**.

### Datasets: (used for evaluation)
1. **Inspec**: 500 English abstracts.
2. **Semeval**: 100 English scientific articles.
3. **DEFT**: 93 French scientific articles.

### Results:
- **Inspec**: Closeness centrality worked best for short documents.
- **Semeval & DEFT**: Degree centrality performed best, comparable to TextRank.

### Limitations:
- Restricts candidates to **nouns and adjectives**, which might overlook other key phrases.
- Only **undirected word graphs** were explored, leaving directed graphs untested.
- Results are specific to benchmark datasets and may not generalize well to other texts or domains.

### Best Use Case:
This methodology is ideal for situations where we don't need to train a model with a curated dataset to perform well for a particular use case.

###Enhanced Graph Weighting Using BERT( code in semantic_graph_textrank.py )
In this implementation, we improved the graph weight assignment algorithm by leveraging contextual embeddings from BERT. Instead of assigning edge weights based solely on the number of co-occurrences between word pairs, we use the cosine similarity between BERT-generated embeddings to capture the semantic relationship between words. This allows for a more nuanced and meaningful representation of the connections between words in the co-occurrence graph. Although we primarily focused on TextRank for keyphrase extraction, this approach enhances the graph formation process, providing a more context-aware framework for extracting key phrases.
