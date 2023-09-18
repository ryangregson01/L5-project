# Sensitivity-Aware Search

## Project Description

Many collections of documents, such as emails, cannot be released to the public because they contain sensitive information. Previous literature has addressed the task of opening collections containing sensitive information by framing the problem as a classification task. However, there has been recent interest in the idea of sensitivity- aware search (SAS) - retrieval models that provide the user with documents relevant to their query, while minimising the number of likely sensitive documents returned.

Previously proposed sensitivity-aware retrieval models have used a learning-to-rank approach and integrated sensitivity features. However, cross-encoder based neural re-rankers have yielded large gains in retrieval effectiveness in many search tasks over recent years. Furthermore, the retrieval performance of such cross-encoder re-rankers can be enhanced by prioritising documents for rescoring that are syntactically similar to those already scored highly by the reranker, a process called graph adaptive reranking (GAR).[1]

In this project, the student will propose and develop a novel sensitivity-aware search retrieval model and evaluate its effectiveness compared to sensitivity-aware search models from the literature and variants of GAR tailored for sensitivity-aware search.

[1] Sean MacAvaney, Nicola Tonellotto, and Craig Macdonald. 2022. Adaptive reranking with a corpus graph. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 1491–1500.

