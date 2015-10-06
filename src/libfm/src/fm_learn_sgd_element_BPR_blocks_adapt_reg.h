// author: Fabio Petroni [www.fabiopetroni.com]
//
// Based on the publication:
// - Fabio Petroni, Luciano Del Corro and Rainer Gemulla (2015):
//   CORE: Context-Aware Open Relation Extraction with Factorization Machines.
//   In Empirical Methods in Natural Language Processing (EMNLP 2015)

#ifndef FM_LEARN_SGD_ELEMENT_BPR_BLOCK_ADAPT_REG_H_
#define FM_LEARN_SGD_ELEMENT_BPR_BLOCK_ADAPT_REG_H_

#include <sstream>
#include "fm_learn_sgd.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include "../../fm_core/fm_sgd.h"
#include <ctime>
#include "BPR_utils.h"
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <math.h>       /* sqrt */
#include "BPRA_model.h"

class fm_learn_sgd_element_BPR_blocks_adapt_reg: public fm_learn_sgd {
	public:

		//THE FOLLOWONG VARIABLES MUST BE SET BEFORE init()
		int num_neg_samples;
		std::string file_out_conv; //optional

		DVector<double> sum_pos, sum_neg;
		std::ofstream out_conv;
		DVector<bool> grad_visited;
		DVector<double> grad;
		DMatrix<double> grad_v;

		bpra_model_class bpra_model;

		virtual void init() {
			fm_learn_sgd::init();
			sum_pos.setSize(fm->num_factor);
			sum_neg.setSize(fm->num_factor);
			if (!file_out_conv.empty()){ out_conv.open(file_out_conv);}
			grad_visited.setSize(fm->num_attribute);
			grad.setSize(fm->num_attribute);
			grad_v.setSize(fm->num_factor, fm->num_attribute);

			bpra_model.num_attribute = fm->num_attribute;
			bpra_model.num_factor = fm->num_factor;
			bpra_model.num_attr_groups = meta->num_attr_groups;
			bpra_model.init();
		}

		virtual void learn(Data& train, Data& test) {
			uint& fixed_block_attr_size = train.relation(FIXED_BLOCK).data->attr_offset;
			//set the model
			fm->reg0 = 0; //always
			for (uint attr_id = 0; attr_id< fixed_block_attr_size; attr_id++){ //only the w for the sampled values are important
				fm->w(attr_id) = 0;
			}

			// start with initial regularization
			for (uint g = 0; g < meta->num_attr_groups; g++) {
				bpra_model.reg_w(g) = fm->regw;
				for (int f = 0; f < fm->num_factor; f++) {
					bpra_model.reg_v(g,f) = fm->regv;
				}
			}

			// make sure that fm-parameters are initialized correctly (no other side effects)
			fm->reg0 = 0;
			fm->regw = 0;
			fm->regv = 0;

			fm_learn::learn(train, test);
			std::cout << "learnrate=" << learn_rate << std::endl;
			std::cout << "learnrates=" << learn_rates(0) << "," << learn_rates(1) << "," << learn_rates(2) << std::endl;
			std::cout << "#iterations=" << num_iter << std::endl;
			std::cout.flush();

			std::cout << "BPR BLOCKS WITH ADAPTIVE REGULARIZATION STARTED ..." << std::endl;

			uint n_samples_cases; //number of rows in the sampled.x file
			uint att_sampled_offsets; //starting offset for the sampled variables = number of columns in the fixed.x file
			n_samples_cases = train.relation(SAMPLED_BLOCK).data->num_cases;
			att_sampled_offsets = train.relation(SAMPLED_BLOCK).data->attr_offset;

			uint& training_samples = train.relation(FIXED_BLOCK).data_row_to_relation_row.dim;
			uint& validation_samples = train.relation(FIXED_BLOCK).validation_data_row_to_relation_row.dim;
			std::cout << "#training samples=" << training_samples << std::endl;
			std::cout << "#validation samples=" << validation_samples << std::endl;

			//pre-step 1 : count positive observations...
			std::map<uint,uint> positive_observations;
			std::map<uint,uint> aux_counter;
			//...in training set...
			for (int i = 0; i<training_samples; i++){
				positive_observations[train.relation(FIXED_BLOCK).data_row_to_relation_row(i)]++;
				aux_counter[train.relation(FIXED_BLOCK).data_row_to_relation_row(i)]++;
			}
			//...in validation set
			for (int i = 0; i<validation_samples; i++){
				positive_observations[train.relation(FIXED_BLOCK).validation_data_row_to_relation_row(i)]++;
				aux_counter[train.relation(FIXED_BLOCK).validation_data_row_to_relation_row(i)]++;
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
			//...in training set...
			for (int i = 0; i<training_samples; i++){
				int n = aux_counter[train.relation(FIXED_BLOCK).data_row_to_relation_row(i)];
				pos_values[train.relation(FIXED_BLOCK).data_row_to_relation_row(i)][n-1] = train.relation(SAMPLED_BLOCK).data_row_to_relation_row(i); //UNIFORM SAMPLING
				aux_counter[train.relation(FIXED_BLOCK).data_row_to_relation_row(i)]--;
			}
			//...in validation set
			for (int i = 0; i<validation_samples; i++){
				int n = aux_counter[train.relation(FIXED_BLOCK).validation_data_row_to_relation_row(i)];
				pos_values[train.relation(FIXED_BLOCK).validation_data_row_to_relation_row(i)][n-1] = train.relation(SAMPLED_BLOCK).validation_data_row_to_relation_row(i); //UNIFORM SAMPLING
				aux_counter[train.relation(FIXED_BLOCK).validation_data_row_to_relation_row(i)]--;
			}
			aux_counter.clear();

			double init_time = time(0);

			// SGD
			std::cout << "#time\tit\tBPR_OPT_THETA [ OBJ - REG ]\tBPR_OPT_LAMBDA";
			if (out_conv){ out_conv << "#time\tit\tBPR_OPT_THETA\tBPR_OPT_LAMBDA"; }
			for (uint g = 0; g < meta->num_attr_groups; g++) {
				std::cout << "\treg_w(" << g <<")\t||reg_v(" << g << ")||";
				if (out_conv){ out_conv << "\treg_w(" << g <<")\t||reg_v(" << g << ")||"; }
			}
			std::cout << std::endl;
			if (out_conv){ out_conv << std::endl; }

			//make an array of indices to shuffle accesses for SGD
			std::vector<int> indices_theta;
			for (int i = 0; i<training_samples; i++){
				indices_theta.push_back(i);
			}
			std::vector<int> indices_lamdba;
			for (int i = 0; i<validation_samples; i++){
				indices_lamdba.push_back(i);
			}

			for (int j = 0; j < num_iter; j++) {
				double total_theta = 0.0;
				double total_lambda = 0.0;
				//suffle array theta
				std::random_shuffle ( indices_theta.begin(), indices_theta.end() );
				for (int pos = 0; pos<training_samples; pos++){
					int i = indices_theta[pos]; //random indices
					//STEP1 - THETA_STEP = BPR-based learning: both lambda and theta are learned
					uint& fixed_block_id = train.relation(FIXED_BLOCK).data_row_to_relation_row(i);
					sparse_row<DATA_FLOAT> &fixed_block = train.relation(FIXED_BLOCK).data->data->get(fixed_block_id);

					sparse_row<DATA_FLOAT> shuf_blocks_pos;
					uint& shuf_block_id = train.relation(SAMPLED_BLOCK).data_row_to_relation_row(i);
					shuf_blocks_pos = train.relation(SAMPLED_BLOCK).data->data->get(shuf_block_id);

					double t_part = 0.0;
					int c_part= 0;
					for (int draw = 0; draw < num_neg_samples; draw++) {
						uint neg_id = drawTagNeg(pos_values[fixed_block_id], positive_observations[fixed_block_id], n_samples_cases);

						sparse_row<DATA_FLOAT> shuf_blocks_neg;
						shuf_blocks_neg = train.relation(SAMPLED_BLOCK).data->data->get(neg_id);

						double score = fm->getDiffScore_blocks(fixed_block, shuf_blocks_pos, shuf_blocks_neg, att_sampled_offsets, sum_pos, sum_neg);
						double normalizer = partial_loss(LOSS_FUNCTION_LN_SIGMOID, score);
						sgd_theta_step(fm, &bpra_model, meta, learn_rate, fixed_block, shuf_blocks_pos, shuf_blocks_neg, normalizer, sum_pos, sum_neg, att_sampled_offsets,grad_visited, grad, grad_v);
						//COMPUTE BPR_OPT
						double sigmoid = 1.0;
						sigmoid /= (1 + exp(-score));
						double LL = std::log(sigmoid);
						t_part += LL;
						c_part += 1;
					}
					total_theta += t_part/c_part;
				}

				//COMPUTE REGULARIZATION
				double Reg = 0;
				for (uint attr=0; attr<fm->num_attribute; attr++){
					uint g = meta->attr_group(attr);
					Reg += bpra_model.reg_w(g)* fm->w(attr) * fm->w(attr);
					for (int f = 0; f < fm->num_factor; f++) {
						Reg += bpra_model.reg_v(g,f) * fm->v(f,attr) * fm->v(f,attr);
					}
				}

				//STEP2- LAMBDA_STEP
				//suffle array lambda
				if (j>0){ //skip first iteration
					std::random_shuffle ( indices_lamdba.begin(), indices_lamdba.end() );
					for (int pos = 0; pos<validation_samples; pos++){
						int i = indices_lamdba[pos]; //random indices
						uint& fixed_block_validation_id = train.relation(FIXED_BLOCK).validation_data_row_to_relation_row(i);
						sparse_row<DATA_FLOAT> &fixed_block_validation = train.relation(FIXED_BLOCK).data->data->get(fixed_block_validation_id);
						sparse_row<DATA_FLOAT> shuf_blocks_validation_pos;
						uint& shuf_block_validation_id = train.relation(SAMPLED_BLOCK).validation_data_row_to_relation_row(i);
						shuf_blocks_validation_pos = train.relation(SAMPLED_BLOCK).data->data->get(shuf_block_validation_id);

						double t_part = 0.0;
						int c_part= 0;
						for (int draw = 0; draw < num_neg_samples; draw++) {
							uint neg_validation_id = drawTagNeg(pos_values[fixed_block_validation_id], positive_observations[fixed_block_validation_id], n_samples_cases);
							sparse_row<DATA_FLOAT> shuf_blocks_validation_neg;
							shuf_blocks_validation_neg = train.relation(SAMPLED_BLOCK).data->data->get(neg_validation_id);

							double score = fm->getDiffScore_blocks(fixed_block_validation, shuf_blocks_validation_pos, shuf_blocks_validation_neg, att_sampled_offsets, sum_pos, sum_neg);
							double normalizer = partial_loss(LOSS_FUNCTION_LN_SIGMOID, score);
							sgd_lambda_step(fm,&bpra_model,meta,learn_rate,fixed_block_validation, shuf_blocks_validation_pos, shuf_blocks_validation_neg, normalizer, att_sampled_offsets);
							//COMPUTE BPR_OPT
							double sigmoid = 1.0;
							sigmoid /= (1 + exp(-score));
							double LL = std::log(sigmoid);
							t_part += LL;
							c_part += 1;
						}
						total_lambda += t_part/c_part;
					}
				}

				//store old values
				if (fm->k1) {
					for (uint i = 0; i<fm->num_attribute; i++){
						bpra_model.old_w(i) = fm->w(i);
					}
				}
				for (uint i = 0; i<fm->num_attribute; i++){
					for (int f = 0; f < fm->num_factor; f++) {
						bpra_model.old_v(f,i) = fm->v(f,i);
					}
				}

				double iteration_time = time(0) - init_time;

				double BPR_OPT = total_theta;
				BPR_OPT -= Reg;

				std::cout << iteration_time << "\t" << j << "\t" << BPR_OPT << " [ " << total_theta << " - " << Reg << " ]\t" <<  total_lambda;
				if (out_conv){ out_conv << iteration_time << "\t" << j << "\t" << BPR_OPT << "\t "<< total_lambda; }

//				//DEBUG ADAPTIVE REGULARIZATION TERMS
//				std::cout << std::endl;
//				for (uint g = 0; g < meta->num_attr_groups; g++) {
//					std::cout << "regw[" << g << "]=" << reg_w(g) << std::endl;
//					for (int f = 0; f < fm->num_factor; f++) {
//						std::cout << "regv[" << g << "," << f << "]=" << reg_v(g,f) << std::endl;
//					}
//				}
//				std::cout << std::endl;

				//PRINT NORM ADAPTIVE REGULARIZATION
				for (uint g = 0; g < meta->num_attr_groups; g++) {
					std::cout << "\t" << bpra_model.reg_w(g);
					if (out_conv){ out_conv << "\t" << bpra_model.reg_w(g); }
					double sum = 0;
					for (int f = 0; f < fm->num_factor; f++) {
						sum += bpra_model.reg_v(g,f)*bpra_model.reg_v(g,f);
					}
					double norm = sqrt(sum);
					std::cout << "\t" << norm;
					if (out_conv){ out_conv << "\t" << norm; }
				}

				std::cout << std::endl;
				if (out_conv){ out_conv << std::endl; }

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
