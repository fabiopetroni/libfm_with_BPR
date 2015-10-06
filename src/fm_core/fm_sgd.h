// Copyright (C) 2010, 2011, 2012, 2013, 2014 Steffen Rendle
// Contact:   srendle@libfm.org, http://www.libfm.org/
//
// This file is part of libFM.
//
// libFM is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// libFM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with libFM.  If not, see <http://www.gnu.org/licenses/>.
//
//
// fm_sgd.h: Generic SGD for elementwise and pairwise losses for Factorization
//           Machines
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.

#ifndef FM_SGD_H_
#define FM_SGD_H_

#include "fm_model.h"

void fm_SGD(fm_model* fm, const double& learn_rate, sparse_row<DATA_FLOAT> &x, const double multiplier, DVector<double> &sum) {
	if (fm->k0) {
		double& w0 = fm->w0; //GLOBAL BIAS
		w0 -= learn_rate * (multiplier + fm->reg0 * w0);
	}
	if (fm->k1) { //USER,ITEMS,ATTRIBUTES,CONTEXT BIAS
		for (uint i = 0; i < x.size; i++) {
			double& w = fm->w(x.data[i].id);
			w -= learn_rate * (multiplier * x.data[i].value + fm->regw * w);
		}
	}	
	for (int f = 0; f < fm->num_factor; f++) {
		for (uint i = 0; i < x.size; i++) {
			double& v = fm->v(f,x.data[i].id);
			double grad = sum(f) * x.data[i].value - v * x.data[i].value * x.data[i].value; 
			v -= learn_rate * (multiplier * grad + fm->regv * v);
		}
	}	
}
		
void fm_pairSGD(fm_model* fm, const double& learn_rate, sparse_row<DATA_FLOAT> &x_pos, sparse_row<DATA_FLOAT> &x_neg, const double multiplier, DVector<double> &sum_pos, DVector<double> &sum_neg, DVector<bool> &grad_visited, DVector<double> &grad) {
	if (fm->k0) {
		double& w0 = fm->w0;
		w0 -= fm->reg0 * w0; // w0 should always be 0
	}

	//UNARY INTERACTION
	if (fm->k1) {

		// STEP 1: build gradients
		for (uint i = 0; i < x_pos.size; i++) {
			grad(x_pos.data[i].id) = 0;
			grad_visited(x_pos.data[i].id) = false;
		}
		for (uint i = 0; i < x_neg.size; i++) {
			grad(x_neg.data[i].id) = 0;
			grad_visited(x_neg.data[i].id) = false;
		}
		for (uint i = 0; i < x_pos.size; i++) {
			grad(x_pos.data[i].id) += x_pos.data[i].value;
		}
		for (uint i = 0; i < x_neg.size; i++) {
			grad(x_neg.data[i].id) -= x_neg.data[i].value;
		}

		// STEP 2: perform stochastic updates
		for (uint i = 0; i < x_pos.size; i++) {
			uint& attr_id = x_pos.data[i].id;
			if (! grad_visited(attr_id)) {
				double& w = fm->w(attr_id);
				w -= learn_rate * (multiplier * grad(attr_id) + fm->regw * w);
				grad_visited(attr_id) = true;
			}
		}
		for (uint i = 0; i < x_neg.size; i++) {
			uint& attr_id = x_neg.data[i].id;
			if (! grad_visited(attr_id)) {
				double& w = fm->w(attr_id);
				w -= learn_rate * (multiplier * grad(attr_id) + fm->regw * w);
				grad_visited(attr_id) = true;
			}
		}
	}

	//PAIRWISE INTERACTIONS
	for (int f = 0; f < fm->num_factor; f++) {
		for (uint i = 0; i < x_pos.size; i++) {
			grad(x_pos.data[i].id) = 0;
			grad_visited(x_pos.data[i].id) = false;
		}
		for (uint i = 0; i < x_neg.size; i++) {
			grad(x_neg.data[i].id) = 0;
			grad_visited(x_neg.data[i].id) = false;
		}
		for (uint i = 0; i < x_pos.size; i++) {
			grad(x_pos.data[i].id) += sum_pos(f) * x_pos.data[i].value - fm->v(f, x_pos.data[i].id) * x_pos.data[i].value * x_pos.data[i].value;
		}
		for (uint i = 0; i < x_neg.size; i++) {
			grad(x_neg.data[i].id) -= sum_neg(f) * x_neg.data[i].value - fm->v(f, x_neg.data[i].id) * x_neg.data[i].value * x_neg.data[i].value;
		}
		for (uint i = 0; i < x_pos.size; i++) {
			uint& attr_id = x_pos.data[i].id;
			if (! grad_visited(attr_id)) {
				double& v = fm->v(f,attr_id);
				v -= learn_rate * (multiplier * grad(attr_id) + fm->regv * v);
				grad_visited(attr_id) = true;
			}
		}
		for (uint i = 0; i < x_neg.size; i++) {
			uint& attr_id = x_neg.data[i].id;
			if (! grad_visited(attr_id)) {
				double& v = fm->v(f,attr_id);
				v -= learn_rate * (multiplier * grad(attr_id) + fm->regv * v);
				grad_visited(attr_id) = true;
			}
		}
	}
}

/*
 * this method performs the stochastic updates for the bpr optimization procedure.
 * It does so using a pairwise approach, with the positive row x (obtained combining fixed_block + shuf_blocks_pos)  and the negative sample x− (obtained combining fixed_block + shuf_blocks_neg)
 * author: Fabio Petroni [www.fabiopetroni.com]
 */
void fm_blocks_pairSGD(fm_model* fm, const double& learn_rate, sparse_row<FM_FLOAT> &fixed_block, sparse_row<FM_FLOAT> &shuf_blocks_pos, sparse_row<FM_FLOAT> &shuf_blocks_neg, const double multiplier, DVector<double> &sum_pos, DVector<double> &sum_neg, uint a_sampled_offsets, DVector<bool> &grad_visited, DVector<double> &grad, DMatrix<double> &grad_v) {
	//UNARY INTERACTION
	if (fm->k1) {
		// STEP 1: build gradients
		for (uint i = 0; i < shuf_blocks_pos.size; i++) {
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
			if (! grad_visited(attr_id_pos)) {
				double& w_pos = fm->w(attr_id_pos);
				double reg = 2 * fm->regw * w_pos;
				w_pos += learn_rate * (multiplier * grad(attr_id_pos) - reg);
				grad_visited(attr_id_pos) = true;
			}
		}
		//negative updates
		for (uint i = 0; i < shuf_blocks_neg.size; i++) {
			uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
			if (! grad_visited(attr_id_neg)) {
				double& w_neg = fm->w(attr_id_neg);
				double reg = 2 * fm->regw * w_neg;
				w_neg += learn_rate * (multiplier * grad(attr_id_neg) - reg);
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
			double& v = fm->v(f,attr_id);
			double grad = diff_sum_neg_pos * fixed_block.data[i].value;
			double reg = 2 * fm->regv * v;
			v += learn_rate * (multiplier * grad - reg);
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
		uint attr_id_pos = shuf_blocks_pos.data[i].id +a_sampled_offsets;
		if (! grad_visited(attr_id_pos)) {
			for (int f = 0; f < fm->num_factor; f++) {
				double& v_pos = fm->v(f,attr_id_pos);
				double reg = 2 * fm->regv * v_pos;
				v_pos += learn_rate * (multiplier * grad_v(f,attr_id_pos) - reg);
			}
			grad_visited(attr_id_pos) = true;
		}
	}
	//negative updates
	for (uint i = 0; i < shuf_blocks_neg.size; i++) {
		uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
		if (! grad_visited(attr_id_neg)) {
			for (int f = 0; f < fm->num_factor; f++) {
				double& v_neg = fm->v(f,attr_id_neg);
				double reg = 2 * fm->regv * v_neg;
				v_neg += learn_rate * (multiplier * grad_v(f,attr_id_neg) - reg);
			}
			grad_visited(attr_id_neg) = true;
		}
	}
}

#endif /*FM_SGD_H_*/
