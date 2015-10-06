// author: Fabio Petroni [www.fabiopetroni.com]
//
// Based on the publication:
// - Fabio Petroni, Luciano Del Corro and Rainer Gemulla (2015):
//   CORE: Context-Aware Open Relation Extraction with Factorization Machines.
//   In Empirical Methods in Natural Language Processing (EMNLP 2015)

#ifndef BPRA_UTILS_H_
#define BPRA_UTILS_H_

#include "../../util/matrix.h"
#include "../../util/fmatrix.h"
#include "BPR_utils.h"

class bpra_model_class {

	public:
		//parameters to set
		int num_attribute;
		int num_factor;
		int num_attr_groups;

		// regularization parameter
		DVector<double> reg_w;
		DMatrix<double> reg_v;

		// for each parameter there is one gradient to store
		DVectorDouble old_w;
		DMatrixDouble old_v;

		// local parameters in the lambda_update step
		DVector<double> lambda_w_grad;
		DVector<double> sum_pos_f;
		DVector<double> sum_neg_f;
		DVector<double> sum_f_dash_f;
		virtual void init() {
			// for each parameter there is an old value to store
			old_w.setSize(num_attribute);
			old_v.setSize(num_factor, num_attribute);

			// regularization parameter
			reg_w.setSize(num_attr_groups);
			reg_v.setSize(num_attr_groups, num_factor);

			// local parameters in the lambda_update step
			lambda_w_grad.setSize(num_attr_groups);
			sum_pos_f.setSize(num_attr_groups);
			sum_neg_f.setSize(num_attr_groups);
			sum_f_dash_f.setSize(num_attr_groups);
		}
};

void sgd_theta_step(fm_model* fm, bpra_model_class* bpra, DataMetaInfo* meta, const double& learn_rate, sparse_row<FM_FLOAT> &fixed_block, sparse_row<FM_FLOAT>& shuf_blocks_pos, sparse_row<FM_FLOAT>& shuf_blocks_neg, const double multiplier, DVector<double> &sum_pos, DVector<double> &sum_neg, uint a_sampled_offsets, DVector<bool> &grad_visited, DVector<double> &grad, DMatrix<double> &grad_v) {
//UNARY INTERACTION
	if (fm->k1) {
		// STEP 1: build gradients
		for (uint i = 0; i <shuf_blocks_pos.size; i++) {
			uint attr_id_pos = shuf_blocks_pos.data[i].id + a_sampled_offsets;
			grad(attr_id_pos) = 0;
			grad_visited(attr_id_pos) = false;
		}
		for (uint i = 0; i < shuf_blocks_neg.size; i++) {
			uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
			grad(attr_id_neg) = 0;
			grad_visited(attr_id_neg) = false;
		}
		for (uint i = 0; i < shuf_blocks_pos.size; i++) {
			uint attr_id_pos = shuf_blocks_pos.data[i].id + a_sampled_offsets;
			grad(attr_id_pos) += shuf_blocks_pos.data[i].value;
		}
		for (uint i = 0; i < shuf_blocks_neg.size; i++) {
			uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
			grad(attr_id_neg) -= shuf_blocks_neg.data[i].value;
		}

		// STEP 2: perform stochastic updates
		//positive updates
		for (uint i = 0; i < shuf_blocks_pos.size; i++) {
			uint attr_id_pos = shuf_blocks_pos.data[i].id + a_sampled_offsets;
			uint g = meta->attr_group(attr_id_pos);
			if (! grad_visited(attr_id_pos)) {
				double& w_pos = fm->w(attr_id_pos);
				double reg = 2 * bpra->reg_w(g) * w_pos;
//					reg /= fm->degree(attr_id_pos);
				w_pos += learn_rate * (multiplier * grad(attr_id_pos) - reg);  //inverted sign by Fabio
				grad_visited(attr_id_pos) = true;
			}
		}
		//negative updates
		for (uint i = 0; i < shuf_blocks_neg.size; i++) {
			uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
			uint g = meta->attr_group(attr_id_neg);
			if (! grad_visited(attr_id_neg)) {
				double& w_neg = fm->w(attr_id_neg);
				double reg = 2 * bpra->reg_w(g) * w_neg;
				w_neg += learn_rate * (multiplier * grad(attr_id_neg) - reg); //inverted sign by Fabio
				grad_visited(attr_id_neg) = true;
			}
		}
	}
	//PAIRWISE INTERACTIONS
	//STEP 0: fixed block updates
	for (int f = 0; f < fm->num_factor; f++) {
		double diff_sum_neg_pos = sum_pos(f) - sum_neg(f);
		for (uint i = 0; i < fixed_block.size; i++) {
			uint attr_id = fixed_block.data[i].id;
			uint g = meta->attr_group(attr_id);
			meta->attr_group(attr_id);
			double& v = fm->v(f,attr_id);
			double grad = diff_sum_neg_pos * fixed_block.data[i].value;
			double reg = 2 * bpra->reg_v(g,f) * v;
//			reg /= fm->degree(attr_id);
			v += learn_rate * (multiplier * grad - reg); //inverted sign by Fabio
		}
	}
	// STEP 1: build gradients
	for (uint i = 0; i < shuf_blocks_pos.size; i++) {
		uint attr_id_pos = shuf_blocks_pos.data[i].id + a_sampled_offsets;
		for (int f = 0; f < fm->num_factor; f++) { grad_v(f,attr_id_pos) = 0; }
		grad_visited(attr_id_pos) = false;
	}
	for (uint i = 0; i < shuf_blocks_neg.size; i++) {
		uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
		for (int f = 0; f < fm->num_factor; f++) { grad_v(f,attr_id_neg) = 0; }
		grad_visited(attr_id_neg) = false;
	}
	for (uint i = 0; i < shuf_blocks_pos.size; i++) {
		uint attr_id_pos = shuf_blocks_pos.data[i].id + a_sampled_offsets;
		for (int f = 0; f < fm->num_factor; f++) { grad_v(f,attr_id_pos) += (sum_pos(f)*shuf_blocks_pos.data[i].value) - (fm->v(f,attr_id_pos)*shuf_blocks_pos.data[i].value*shuf_blocks_pos.data[i].value); }
	}
	for (uint i = 0; i < shuf_blocks_neg.size; i++) {
		uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
		for (int f = 0; f < fm->num_factor; f++) { grad_v(f,attr_id_neg) -= (sum_neg(f)*shuf_blocks_neg.data[i].value) - (fm->v(f,attr_id_neg)*shuf_blocks_neg.data[i].value *shuf_blocks_neg.data[i].value); }
	}
	// STEP 2: perform stochastic updates
	//positive updates
	for (uint i = 0; i < shuf_blocks_pos.size; i++) {
		uint attr_id_pos = shuf_blocks_pos.data[i].id + a_sampled_offsets;
		if (! grad_visited(attr_id_pos)) {
			uint g = meta->attr_group(attr_id_pos);
			for (int f = 0; f < fm->num_factor; f++) {
				double& v_pos = fm->v(f,attr_id_pos);
				double reg = 2 * bpra->reg_v(g,f) * v_pos;
//					reg /= fm->degree(attr_id_pos);
				v_pos += learn_rate * (multiplier * grad_v(f,attr_id_pos) - reg); //inverted sign by Fabio
			}
			grad_visited(attr_id_pos) = true;
		}
	}
	//negative updates
	for (uint i = 0; i < shuf_blocks_neg.size; i++) {
		uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
		if (! grad_visited(attr_id_neg)) {
			uint g = meta->attr_group(attr_id_neg);
			for (int f = 0; f < fm->num_factor; f++) {
				double& v_neg = fm->v(f,attr_id_neg);
				double reg = 2 * bpra->reg_v(g,f) * v_neg;
				v_neg += learn_rate * (multiplier * grad_v(f,attr_id_neg) - reg); //inverted sign by Fabio
			}
			grad_visited(attr_id_neg) = true;
		}
	}
}

void sgd_lambda_step(fm_model* fm, bpra_model_class* bpra, DataMetaInfo* meta, const double& learn_rate, sparse_row<FM_FLOAT> &fixed_block, sparse_row<FM_FLOAT> &shuf_blocks_pos, sparse_row<FM_FLOAT> &shuf_blocks_neg, const double multiplier, uint a_sampled_offsets) {
	//UNARY INTERACTION REGULARIZATION VALUES
	if (fm->k1) {
		bpra->lambda_w_grad.init(0.0);
		// STEP 1: build gradients
		for (uint i = 0; i < shuf_blocks_pos.size; i++) {
			uint attr_id_pos = shuf_blocks_pos.data[i].id + a_sampled_offsets;
			uint g = meta->attr_group(attr_id_pos);
			bpra->lambda_w_grad(g) += shuf_blocks_pos.data[i].value * bpra->old_w(attr_id_pos);
		}
		for (uint i = 0; i < shuf_blocks_neg.size; i++) {
			uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
			uint g = meta->attr_group(attr_id_neg);
			bpra->lambda_w_grad(g) -= shuf_blocks_neg.data[i].value * bpra->old_w(attr_id_neg);
		}
		// STEP 2: perform stochastic updates
		for (uint g = 0; g < meta->num_attr_groups; g++) {
			bpra->lambda_w_grad(g) = -2 * learn_rate * bpra->lambda_w_grad(g);
			bpra->reg_w(g) += learn_rate * multiplier * bpra->lambda_w_grad(g);
			bpra->reg_w(g) = std::max(0.0, bpra->reg_w(g));
		}
	}
	//PAIRWISE INTERACTIONS REGULARIZATION VALUES
	for (int f = 0; f < fm->num_factor; f++) {
		double sum_pos_f_dash = 0; //this is independent of the groups
		double sum_neg_f_dash = 0; //this is independent of the groups
		bpra->sum_pos_f.init(0.0);
		bpra->sum_neg_f.init(0.0);
		bpra->sum_f_dash_f.init(0.0);
		// STEP 1: build gradients
		for (uint i = 0; i < shuf_blocks_pos.size; i++) {
			uint attr_id_pos = shuf_blocks_pos.data[i].id + a_sampled_offsets;
			uint g = meta->attr_group(attr_id_pos);
			double& v = fm->v(f,attr_id_pos);
			sum_pos_f_dash += bpra->old_v(f,attr_id_pos) * shuf_blocks_pos.data[i].value;
			bpra->sum_pos_f(g) += v * shuf_blocks_pos.data[i].value;
			bpra->sum_f_dash_f(g) += bpra->old_v(f,attr_id_pos) * shuf_blocks_pos.data[i].value * v * shuf_blocks_pos.data[i].value;
		}
		for (uint i = 0; i < shuf_blocks_neg.size; i++) {
			uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
			uint g = meta->attr_group(attr_id_neg);
			double& v = fm->v(f,attr_id_neg);
			sum_neg_f_dash += bpra->old_v(f,attr_id_neg) * shuf_blocks_neg.data[i].value;
			bpra->sum_neg_f(g) += v * shuf_blocks_neg.data[i].value;
			bpra->sum_f_dash_f(g) -= bpra->old_v(f,attr_id_neg) * shuf_blocks_neg.data[i].value * v * shuf_blocks_neg.data[i].value;
		}
		// STEP 2: perform stochastic updates
		for (uint g = 0; g < meta->num_attr_groups; g++) {
			double lambda_v_grad = -2 * learn_rate *  ( (sum_pos_f_dash * bpra->sum_pos_f(g)) - (sum_neg_f_dash * bpra->sum_neg_f(g)) - bpra->sum_f_dash_f(g) );
			bpra->reg_v(g,f) += learn_rate * multiplier * lambda_v_grad;
			bpra->reg_v(g,f) = std::max(0.0, bpra->reg_v(g,f));
		}
	}
}
#endif /*BPRA_UTILS_H_*/
