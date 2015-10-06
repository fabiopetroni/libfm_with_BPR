// author: Fabio Petroni [www.fabiopetroni.com]
//
// Based on the publication:
// - Fabio Petroni, Luciano Del Corro and Rainer Gemulla (2015):
//   CORE: Context-Aware Open Relation Extraction with Factorization Machines.
//   In Empirical Methods in Natural Language Processing (EMNLP 2015)

#ifndef BPR_UTILS_H_
#define BPR_UTILS_H_

#define FIXED_BLOCK	0
#define SAMPLED_BLOCK	1

#include <assert.h>

uint drawTagNeg(uint* pos_ids, uint& pos_observations, uint size) {
	uint t_n;
	do {
		t_n = (rand() % size);
	} while (std::find(pos_ids, pos_ids+pos_observations, t_n)!=pos_ids+pos_observations); //t_n in pos_ids, redo iteration
	return t_n;
}

double partial_loss(int loss_function, double x) {
	if (loss_function == LOSS_FUNCTION_SIGMOID) {
				double sigmoid_tp_tn = (double) 1/(1+exp(-x));
				return sigmoid_tp_tn*(1-sigmoid_tp_tn);
	} else if (loss_function == LOSS_FUNCTION_LN_SIGMOID) {
		double exp_x = exp(-x);
			return exp_x / (1 + exp_x);
		} else {
		assert((loss_function == LOSS_FUNCTION_LN_SIGMOID) || (loss_function == LOSS_FUNCTION_SIGMOID));
	}
	return 0;
}
#endif /*BPR_UTILS_H_*/
