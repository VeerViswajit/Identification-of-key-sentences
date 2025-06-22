# 📚 Key Sentence Identification for Educational Texts

This project introduces the **Key Sentence Identifier (KSI)** — a domain-specific NLP pipeline designed to extract contextually relevant key sentences from student responses in the **Operating Systems** domain. The KSI model combines fine-tuned BERT for token-level keyword extraction and SBERT for sentence-level classification, enhanced with dynamic thresholding and embedding fusion to accurately identify core concepts. A custom dataset with annotated sentences mimicking real-world student answers supports the training and evaluation.

Alongside the KSI implementation, this repository also includes re-implementations of several key approaches from related literature. The **Centrality Measures** module explores the role of **degree centrality** in identifying important sentences. The **Seq2Seq model** captures sequential dependencies in a supervised learning setting for keyphrase generation. The **Transformer-based methods** replicate extractive techniques that rely on pre-trained language models to evaluate sentence-level importance, inspired by recent advances in sentence ranking and summarization.

Together, these implementations provide a comparative foundation for understanding different strategies in educational content analysis, while the KSI model stands as the project's primary contribution.
