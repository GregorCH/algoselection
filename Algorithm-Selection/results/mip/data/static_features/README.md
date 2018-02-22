In this directory you can find all data produced by prediction models.

Legend:
	a - Results of Alex's prediction models.
	b - Results of Bart's prediction models.

Currently in use are:
	a_scaled	- Data used for training models was scaled ... EXPLAIN HOW
	b_all_features - All models are trained on full set of features.


Archive directory contains data produced by models that are no longer
in use because their performance was not the best.
List of archived models:

	a_nonscaled - Data used for training was not scaled.
	a_scaled_type1 - Data used for training was scaled ... EXPLAIN HOW
	a_scaled_type3 - Data used fot training was scaled ... EXPLAIN HOW
	(Trained on 80%, tested on 20%)

	b_all_features - All classifiers trained on full set of features.
	b_pca_71_features - All classifiers trained on 71 PCA features.
	b_pca_features - All classifiers trained just on PCA features.
	b_reduced_features - Classifiers trained on reduced set of features ... EXPLAIN HOW IS THE SET PRODUCED.
	(Trained on 90%, tested on 10%)
