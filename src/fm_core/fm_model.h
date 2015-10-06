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
// fm_model.h: Model for Factorization Machines
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.

#ifndef FM_MODEL_H_
#define FM_MODEL_H_

#include "../util/matrix.h"
#include "../util/fmatrix.h"
#include "fm_data.h"
#include <iostream>
#include <stdlib.h>     /* atof */
#include <fstream>
#include <string>
#include <vector>
#include "../libfm/src/Data.h"

#define FIXED_BLOCK	0


class fm_model {
	private:
		DVector<double> m_sum, m_sum_sqr;
	public:
		double w0;
		DVectorDouble w;
		DMatrixDouble v;

	public:
		// the following values should be set:
		uint num_attribute;
		
		bool k0, k1;
		int num_factor;
		
		double reg0;
		double regw, regv;
		
		double init_stdev;
		double init_mean;
		
		fm_model();
		void debug();
		void init();
		double predict(sparse_row<FM_FLOAT>& x);
		double predict(sparse_row<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr);

		//methods added by Fabio Petroni [www.fabiopetroni.com] to integrate bpr inside libfm
		double getDiffScore_blocks(sparse_row<FM_FLOAT> &fixed_block, sparse_row<FM_FLOAT> &shuf_blocks_pos, sparse_row<FM_FLOAT> &shuf_blocks_neg, uint a_sampled_offsets, DVector<double> &sum_pos, DVector<double> &sum_neg);
		double predict_case_blocks(sparse_row<FM_FLOAT> &fixed_block, sparse_row<FM_FLOAT> &shuf_blocks, uint a_sampled_offsets);
		void resumeState(std::string vectors_file_name);
		void printOutState(std::string vectors_file_name);
	private:
		void split(const std::string& s, char c, std::vector<std::string>& v);
};

fm_model::fm_model() {
	num_factor = 0;
	init_mean = 0;
	init_stdev = 0.01;
	reg0 = 0.0;
	regw = 0.0;
	regv = 0.0; 
	k0 = true;
	k1 = true;
}

void fm_model::debug() {
	std::cout << "num_attributes=" << num_attribute << std::endl;
	std::cout << "use w0=" << k0 << std::endl;
	std::cout << "use w1=" << k1 << std::endl;
	std::cout << "dim v =" << num_factor << std::endl;
	std::cout << "reg_w0=" << reg0 << std::endl;
	std::cout << "reg_w=" << regw << std::endl;
	std::cout << "reg_v=" << regv << std::endl; 
	std::cout << "init ~ N(" << init_mean << "," << init_stdev << ")" << std::endl;
}

void fm_model::init() {
	w0 = 0;
	w.setSize(num_attribute);
	v.setSize(num_factor, num_attribute);
	w.init(0);
	v.init(init_mean, init_stdev);
	m_sum.setSize(num_factor);
	m_sum_sqr.setSize(num_factor);
}

double fm_model::predict(sparse_row<FM_FLOAT>& x) {
	return predict(x, m_sum, m_sum_sqr);		
}

double fm_model::predict(sparse_row<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr) {
	double result = 0;
	if (k0) {	
		result += w0;
	}
	if (k1) {
		for (uint i = 0; i < x.size; i++) {
			assert(x.data[i].id < num_attribute);
			result += w(x.data[i].id) * x.data[i].value;
		}
	}
	for (int f = 0; f < num_factor; f++) {
		sum(f) = 0;
		sum_sqr(f) = 0;
		for (uint i = 0; i < x.size; i++) {
			double d = v(f,x.data[i].id) * x.data[i].value;
			sum(f) += d;
			sum_sqr(f) += d*d;
		}
		result += 0.5 * (sum(f)*sum(f) - sum_sqr(f));
	}
	return result;
}

/*
 * this method return the difference d(x, x−) between the score of the positive row x (obtained combining fixed_block + shuf_blocks_pos)  and the negative sample x− (obtained combining fixed_block + shuf_blocks_neg)
 * author: Fabio Petroni [www.fabiopetroni.com]
 */
double fm_model::getDiffScore_blocks(sparse_row<FM_FLOAT> &fixed_block, sparse_row<FM_FLOAT> &shuf_blocks_pos, sparse_row<FM_FLOAT> &shuf_blocks_neg, uint a_sampled_offsets, DVector<double> &sum_pos, DVector<double> &sum_neg) {
	double result = 0;
	if (k1) {
		//positive sampled updates
		for (uint i = 0; i < shuf_blocks_pos.size; i++) {
			uint attr_id_pos = shuf_blocks_pos.data[i].id + a_sampled_offsets;
			assert(attr_id_pos < num_attribute);
			result += w(attr_id_pos) * shuf_blocks_pos.data[i].value;
		}
		//negative sampled updates
		for (uint i = 0; i < shuf_blocks_neg.size; i++) {
			uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
			assert(attr_id_neg < num_attribute);
			result -= w(attr_id_neg) * shuf_blocks_neg.data[i].value;
		}
	}
	for (int f = 0; f < num_factor; f++) {
		sum_pos(f) = 0;
		sum_neg(f) = 0;
		double sum_sqr_pos = 0;
		double sum_sqr_neg = 0;
		//positive sampled updates
		for (uint i = 0; i < shuf_blocks_pos.size; i++) {
			uint attr_id_pos = shuf_blocks_pos.data[i].id + a_sampled_offsets;
			double d = v(f,attr_id_pos) * shuf_blocks_pos.data[i].value;
			sum_pos(f) += d;
			sum_sqr_pos += d*d;
		}
		//negative sampled updates
		for (uint i = 0; i < shuf_blocks_neg.size; i++) {
			uint attr_id_neg = shuf_blocks_neg.data[i].id + a_sampled_offsets;
			double d = v(f,attr_id_neg) * shuf_blocks_neg.data[i].value;
			sum_neg(f) += d;
			sum_sqr_neg += d*d;
		}
		//fixed updates
		for (uint i = 0; i < fixed_block.size; i++) {
			double d = v(f,fixed_block.data[i].id) * fixed_block.data[i].value;
			sum_pos(f) += d;
			sum_neg(f) += d;
		}
		result += 0.5 * (sum_pos(f)*sum_pos(f) - sum_sqr_pos);
		result -= 0.5 * (sum_neg(f)*sum_neg(f) - sum_sqr_neg);
	}
	return result;
}

/*
 * this method return the the score of a row x, obtained combining fixed_block + shuf_blocks
 * author: Fabio Petroni [www.fabiopetroni.com]
 */
double fm_model::predict_case_blocks(sparse_row<FM_FLOAT> &fixed_block, sparse_row<FM_FLOAT> &shuf_blocks, uint a_sampled_offsets) {
	double result = 0;
	if (k0) {
		result += w0;
	}
	if (k1) {
		//fixed block
		for (uint i = 0; i < fixed_block.size; i++) {
			uint attr_id = fixed_block.data[i].id;
			assert(attr_id < num_attribute);
			result += w(attr_id) * fixed_block.data[i].value;
		}
		//sampled_block
		for (uint i = 0; i < shuf_blocks.size; i++) {
			uint attr_id = shuf_blocks.data[i].id + a_sampled_offsets;
			assert(attr_id < num_attribute);
			result += w(attr_id) * shuf_blocks.data[i].value;
		}
	}
	for (int f = 0; f < num_factor; f++) {
		double sum = 0;
		double sum_sqr = 0;
		//fixed block
		for (uint i = 0; i < fixed_block.size; i++) {
			uint attr_id = fixed_block.data[i].id;
			double d = v(f,attr_id) * fixed_block.data[i].value;
			sum += d;
			sum_sqr += d*d;
		}
		//sampled_block
		for (uint i = 0; i < shuf_blocks.size; i++) {
			uint attr_id = shuf_blocks.data[i].id + a_sampled_offsets;
			double d = v(f,attr_id) * shuf_blocks.data[i].value;
			sum += d;
			sum_sqr += d*d;
		}
		result += 0.5*(sum*sum-sum_sqr);
	}
	return result;
}

/*
 * this method print in a file (path specified in vectors_file_name) the vectors of all the variables.
 * Notice that such file can be given as input to libfm to resume a computation.
 * author: Fabio Petroni [www.fabiopetroni.com]
 */
void fm_model::printOutState(std::string vectors_file_name){
	std::ofstream out_vectors;
	out_vectors.open(vectors_file_name);
	if (k0) {
		out_vectors << "#global bias W0" << std::endl;
		out_vectors << w0 << std::endl;
	}
	if (k1) {
		out_vectors << "#unary interactions Wj" << std::endl;
		for (uint i = 0; i<num_attribute; i++){
			out_vectors <<	w(i) << std::endl;
		}
	}
	out_vectors << "#pairwise interactions Vj,f" << std::endl;
	for (uint i = 0; i<num_attribute; i++){
		for (int f = 0; f < num_factor; f++) {
			out_vectors << v(f,i) << ' ';
		}
		out_vectors << std::endl;
	}
	out_vectors.close();
}

/*
 * this method read from a file (path specified in vectors_file_name) the vectors of all the variables.
 * The initial state for the libfm will be resumed from the stored state.
 * author: Fabio Petroni [www.fabiopetroni.com]
 */
void fm_model::resumeState(std::string vectors_file_name) {
	std::string line;
	std::ifstream myfile (vectors_file_name);
	if (myfile.is_open()){
		if (k0) {
			std::getline(myfile,line); // "#global bias W0"
			std::getline(myfile,line);
			w0 = std::atof(line.c_str());
		}
		if (k1) {
			std::getline(myfile,line); //"#unary interactions Wj" << std::endl
			for (uint i = 0; i<num_attribute; i++){
				std::getline(myfile,line);
				w(i) = std::atof(line.c_str());
			}
		}
		std::getline(myfile,line); // "#pairwise interactions Vj,f"
		for (uint i = 0; i<num_attribute; i++){
			std::getline(myfile,line);
			std::vector<std::string> v_str;
			split(line, ' ', v_str);
			for (int f = 0; f < num_factor; f++) {
				v(f,i) = std::atof(v_str[f].c_str());
			}
		}
		myfile.close();
	}
	else throw "Unable to open file with vectors to resume state";
}

/*
 * util method
 * author: Fabio Petroni [www.fabiopetroni.com]
 */
void fm_model::split(const std::string& s, char c, std::vector<std::string>& v) {
	std::string::size_type i = 0;
	std::string::size_type j = s.find(c);
	while (j != std::string::npos) {
		v.push_back(s.substr(i, j-i));
		i = ++j;
		j = s.find(c, j);
		if (j == std::string::npos)
			v.push_back(s.substr(i, s.length()));
	}
}

#endif /*FM_MODEL_H_*/
