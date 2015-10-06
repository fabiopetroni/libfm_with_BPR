// author: Fabio Petroni [www.fabiopetroni.com]
//
// Based on the publication:
// - Fabio Petroni, Luciano Del Corro and Rainer Gemulla (2015):
//   CORE: Context-Aware Open Relation Extraction with Factorization Machines.
//   In Empirical Methods in Natural Language Processing (EMNLP 2015)

#ifndef FM_LEARN_SGD_ELEMENT_BPR_BLOCK_H_
#define FM_LEARN_SGD_ELEMENT_BPR_BLOCK_H_

#include "fm_learn_sgd.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include "../../fm_core/fm_sgd.h"
#include <ctime>
#include "BPR_utils.h"
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector

class fm_learn_sgd_element_BPR_blocks: public fm_learn_sgd {
	public:

		//VARIABLES TO BE SET BEFORE init()
		int num_neg_samples;
		std::string file_out_conv; //optional

		//AUXILIAR VARIABLES
		DVector<double> sum_pos, sum_neg;
		std::ofstream out_conv;
		DVector<bool> grad_visited;
		DVector<double> grad;
		DMatrix<double> grad_v;

		virtual void init() {
			fm_learn_sgd::init();
			sum_pos.setSize(fm->num_factor);
			sum_neg.setSize(fm->num_factor);
			if (!file_out_conv.empty()){ out_conv.open(file_out_conv);}
			grad_visited.setSize(fm->num_attribute);
			grad.setSize(fm->num_attribute);
			grad_v.setSize(fm->num_factor, fm->num_attribute);
		}

		virtual void learn(Data& train, Data& test) {

			uint& fixed_block_attr_size = train.relation(FIXED_BLOCK).data->attr_offset;

			//set the model
			fm->reg0 = 0; //always
			for (uint attr_id = 0; attr_id< fixed_block_attr_size; attr_id++){ //only the w for the sampled values are important
				fm->w(attr_id) = 0;
			}

			fm_learn::learn(train, test);
			std::cout << "learnrate=" << learn_rate << std::endl;
			std::cout << "learnrates=" << learn_rates(0) << "," << learn_rates(1) << "," << learn_rates(2) << std::endl;
			std::cout << "#iterations=" << num_iter << std::endl;
			std::cout.flush();
			std::cout << "BPR BLOCKS STARTED ..." << std::endl;

			//some checks
			if (train.relation.dim != 2 ){
				throw "ERROR, the ranked list otput can be computed only if the number of blocks is equal to two";
			}
			if (train.relation(FIXED_BLOCK).data_row_to_relation_row.dim != train.relation(SAMPLED_BLOCK).data_row_to_relation_row.dim){
				throw "ERROR, blocks must have the same number of training points!";
			}
			uint n_samples_cases; //number of rows in the sampled.x file
			uint att_sampled_offsets; //starting offset for the sampled variables = number of columns in the fixed.x file
			n_samples_cases = train.relation(SAMPLED_BLOCK).data->num_cases;
			att_sampled_offsets = train.relation(SAMPLED_BLOCK).data->attr_offset;
			uint& training_samples = train.relation(FIXED_BLOCK).data_row_to_relation_row.dim;
			//pre-step 1 : count positive observations
			std::map<uint,uint> pos_observations;
			std::map<uint,uint> aux_counter;
			for (int i = 0; i<training_samples; i++){
				pos_observations[train.relation(FIXED_BLOCK).data_row_to_relation_row(i)]++;
				aux_counter[train.relation(FIXED_BLOCK).data_row_to_relation_row(i)]++;
			}
			//pre-step 2 : build arrays
			std::map<uint,uint*> pos_values;
			std::map<uint,uint>::iterator map_iter;
			for(map_iter=aux_counter.begin(); map_iter!=aux_counter.end(); ++map_iter){
				pos_values[map_iter->first] = new uint[map_iter->second];
				for (int i = 0; i< map_iter->second; i++){
					pos_values[map_iter->first][i] = 0;
				}
			}
			//pre-step 3 : populate arrays
			for (int i = 0; i<training_samples; i++){
				int n = aux_counter[train.relation(FIXED_BLOCK).data_row_to_relation_row(i)];
				pos_values[train.relation(FIXED_BLOCK).data_row_to_relation_row(i)][n-1] = train.relation(SAMPLED_BLOCK).data_row_to_relation_row(i); //UNIFORM SAMPLING
				aux_counter[train.relation(FIXED_BLOCK).data_row_to_relation_row(i)]--;
			}
			aux_counter.clear();

//			//DEBUG POSITIVE OBSERVATIONS MAP
//			std::map<uint,uint*>::iterator map_iter_debug;
//			for(map_iter_debug=pos_values.begin(); map_iter_debug!=pos_values.end(); ++map_iter_debug){
//				std::cout << map_iter_debug->first << " {";
//				for (int i = 0; i<pos_observations[map_iter_debug->first]; i++){
//					std::cout << " " << map_iter_debug->second[i];
//				}
//				std::cout << " }" << std::endl;
//			}

			double init_time = time(0);

			// SGD
			std::cout << "#time\tit\tBPR_OPT" << std::endl;
			if (out_conv){ out_conv << "#time\tit\tBPR_OPT" << std::endl; }

			//make an array of indices to shuffle accesses for SGD
			std::vector<int> indices;
			for (int i = 0; i<training_samples; i++){
				indices.push_back(i);
			}

			for (int j = 0; j < num_iter; j++) {
				double total = 0;
				int count = 0;

				//suffle array
				std::random_shuffle ( indices.begin(), indices.end() );

				for (int pos = 0; pos<training_samples; pos++){

					int i = indices[pos]; //random indices

					double t_part = 0.0;
					int c_part= 0;
					uint& fixed_block_id = train.relation(FIXED_BLOCK).data_row_to_relation_row(i);
					sparse_row<DATA_FLOAT> &fixed_block = train.relation(FIXED_BLOCK).data->data->get(fixed_block_id);

					sparse_row<DATA_FLOAT> shuf_blocks_pos;
					uint& shuf_block_id = train.relation(SAMPLED_BLOCK).data_row_to_relation_row(i);
					shuf_blocks_pos = train.relation(SAMPLED_BLOCK).data->data->get(shuf_block_id);

					for (int draw = 0; draw < num_neg_samples; draw++) {
						uint neg_id = drawTagNeg(pos_values[fixed_block_id], pos_observations[fixed_block_id], n_samples_cases);

//						//DEBUG NEGATIVE SAMPLE
//						std::cout << fixed_block_id << " [";
//						std::cout << " " << neg_id;
//						std::cout << " ]" << std::endl;

						sparse_row<DATA_FLOAT> shuf_blocks_neg;
						shuf_blocks_neg = train.relation(SAMPLED_BLOCK).data->data->get(neg_id);

						double score = fm->getDiffScore_blocks(fixed_block, shuf_blocks_pos, shuf_blocks_neg, att_sampled_offsets, sum_pos, sum_neg);

						double normalizer = partial_loss(LOSS_FUNCTION_LN_SIGMOID, score);

						fm_blocks_pairSGD(fm, learn_rate, fixed_block, shuf_blocks_pos, shuf_blocks_neg, normalizer, sum_pos, sum_neg, att_sampled_offsets, grad_visited, grad, grad_v);

						double LL = std::log(1 / (1 + exp(-score)));
						t_part += LL;
						c_part += 1;
					}
					total += t_part/c_part;
					count += 1;
				}
				double iteration_time = time(0) - init_time;

				//COMPUTE REGULARIZATION
				double Reg = 0;
				for (uint attr=0; attr<fm->num_attribute; attr++){
					Reg += fm->regw * fm->w(attr) * fm->w(attr);
					double sum = 0;
					for (int f = 0; f < fm->num_factor; f++) {
						sum += fm->v(f,attr);
					}
					Reg += fm->regv * sum * sum;
				}

				double BPR_OPT = total;
				BPR_OPT -= Reg;

				std::cout << iteration_time << "\t" << j << "\t" << BPR_OPT << std::endl;
				if (out_conv){ out_conv << iteration_time << "\t" << j << "\t" << BPR_OPT << std::endl; }
			}
			if (out_conv){ out_conv.close(); }
		}

		void predict(Data& data, DVector<double>& out) {
			assert(data.data->getNumRows() == out.dim);
			uint att_sampled_offsets = data.relation(SAMPLED_BLOCK).data->attr_offset;
			uint& test_samples = data.relation(FIXED_BLOCK).data_row_to_relation_row.dim;
			for (int i = 0; i<test_samples; i++){
				uint& fixed_block_id = data.relation(FIXED_BLOCK).data_row_to_relation_row(i);
				sparse_row<DATA_FLOAT> &fixed_block = data.relation(FIXED_BLOCK).data->data->get(fixed_block_id);
				uint& shuf_block_id = data.relation(SAMPLED_BLOCK).data_row_to_relation_row(i);
				sparse_row<DATA_FLOAT> shuf_blocks = data.relation(SAMPLED_BLOCK).data->data->get(shuf_block_id);
				double p = fm->predict_case_blocks(fixed_block,shuf_blocks,att_sampled_offsets);
				out(i) = p;
			}
		}
};

#endif /*FM_LEARN_SGD_ELEMENT_BPR_H_*/
