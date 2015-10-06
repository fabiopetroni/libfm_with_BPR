// author: Fabio Petroni [www.fabiopetroni.com]
//
// Based on the publication:
// - Fabio Petroni, Luciano Del Corro and Rainer Gemulla (2015):
//   CORE: Context-Aware Open Relation Extraction with Factorization Machines.
//   In Empirical Methods in Natural Language Processing (EMNLP 2015)

#ifndef LIBFM_SRC_FM_LEARN_SGD_ELEMENT_BPR_BLOCKS_COMPUTE_RANKED_LIST_PARALLEL_H_
#define LIBFM_SRC_FM_LEARN_SGD_ELEMENT_BPR_BLOCKS_COMPUTE_RANKED_LIST_PARALLEL_H_

#include <vector>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <algorithm>
#include "BPR_utils.h"
#include <iostream>
#include <fstream>

///**** init Multithreading ****/

struct WeightedTag {
	int tag_id;
	double weight;
};
bool sortByWeight(const WeightedTag &lhs, const WeightedTag &rhs) { return lhs.weight < rhs.weight; }

struct thread_data_rl {
   Data train;
   uint fixed_block_id; //fixed block element id to compute ranked list for
   uint num_sampled_cases;
   uint a_sampled_offsets;
   uint pos_observations;
   uint* pos_values;
   fm_model* fm;
   uint TOP_K;
};

void *computeRankedList(void *threadarg){
	struct thread_data_rl *t_data;
	t_data = (struct thread_data_rl *) threadarg;
	sparse_row<DATA_FLOAT> fixed_block = t_data->train.relation(FIXED_BLOCK).data->data->get(t_data->fixed_block_id);
	uint size = t_data->num_sampled_cases - t_data->pos_observations;
	WeightedTag* weighted_tag = new WeightedTag[size];
	WeightedTag* weighted_tag_result = new WeightedTag[std::min(t_data->TOP_K,size)];
	int counter = 0;
	for (int i = 0; i< t_data->num_sampled_cases; i++){
		if (std::find(t_data->pos_values, t_data->pos_values+t_data->pos_observations, i) == t_data->pos_values+t_data->pos_observations){
			uint& sampled_block_id = t_data->train.relation(SAMPLED_BLOCK).data_row_to_relation_row(i);
			sparse_row<DATA_FLOAT> sampled_block =  t_data->train.relation(SAMPLED_BLOCK).data->data->get(sampled_block_id);
			double p = t_data->fm->predict_case_blocks(fixed_block,sampled_block,t_data->a_sampled_offsets);
			weighted_tag[counter].tag_id = i;
			weighted_tag[counter].weight = p;
			counter++;
		}
	}
	std::sort(weighted_tag, weighted_tag+size,sortByWeight);
	for (int t = 0; t < std::min(t_data->TOP_K,size); t++) {
		int tag = weighted_tag[size-t-1].tag_id;
		double w = weighted_tag[size-t-1].weight;
		weighted_tag_result[t].tag_id = tag;
		weighted_tag_result[t].weight = w;
	}
	pthread_exit(weighted_tag_result);
}

class Recommendation {
	public:
		//parameters to be set
		fm_model* fm;
		std::vector<int> target_ids;
		int MAX_THREADS;
		int TOP_K;
		std::string OUT_DIR;

		void evaluate(Data& train);
};

void Recommendation::evaluate(Data& train) {
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
//	//DEBUG POSITIVE OBSERVATIONS MAP
//	std::map<uint,uint*>::iterator map_iter_debug;
//	for(map_iter_debug=pos_values.begin(); map_iter_debug!=pos_values.end(); ++map_iter_debug){
//		std::cout << map_iter_debug->first << " {";
//		for (int i = 0; i<pos_observations[map_iter_debug->first]; i++){
//			std::cout << " " << map_iter_debug->second[i];
//		}
//		std::cout << " }" << std::endl;
//	}
	int n_fixed_cases = train.relation(FIXED_BLOCK).data->num_cases;
	int N = target_ids.size();
	int index = 0;
	while (N > 0){
		int NUM_THREADS = std::min(N,MAX_THREADS);
		pthread_t threads_computeRankedList[NUM_THREADS];
		struct thread_data_rl* td = new thread_data_rl[NUM_THREADS];
		for (int t = 0; t < NUM_THREADS; t++ ){
			int id = target_ids[index];
			assert (id < n_fixed_cases);
			index++;
			td[t].fixed_block_id = id;
			td[t].train = train;
			td[t].num_sampled_cases = n_samples_cases;
			td[t].pos_observations = pos_observations[id];
			td[t].pos_values = pos_values[id];
			td[t].a_sampled_offsets = att_sampled_offsets;
			td[t].fm = fm;
			td[t].TOP_K = TOP_K;
			int rc = pthread_create(&threads_computeRankedList[t], NULL, computeRankedList, (void *)&td[t]);
			if (rc){
				std::cout << "Error:unable to create thread," << rc << std::endl;
				exit(-1);
			}
		}
		void* weighted_tag_result;
		std::ofstream out_books;
		for(int t=0; t < NUM_THREADS; t++ ){
			int rc = pthread_join(threads_computeRankedList[t], (void**)&weighted_tag_result);
			if (rc){
				std::cout << "Error:unable to join," << rc << std::endl;
				exit(-1);
			}
			std::string file_name = OUT_DIR + "/";
			file_name = file_name + std::to_string(td[t].fixed_block_id);
			file_name = file_name + ".dat";
//			std::cout << file_name << std::endl;
//			std::cout << td[t].fixed_block_id << ": " << std::endl;
			std::ofstream out;
			out.open(file_name.c_str());
			int size = td[t].num_sampled_cases - td[t].pos_observations;
			uint K = std::min(TOP_K, size);
			for (uint k = 0; k< K; k++){
				int variable_id = ((WeightedTag*)weighted_tag_result)[k].tag_id;
				double score = ((WeightedTag*)weighted_tag_result)[k].weight;
//				std::cout << variable_id << "\t" << score << std::endl;
				out << variable_id << "\t" << score << std::endl;
			}
			out.close();
		}
		N -= NUM_THREADS;
	}
}

#endif /* LIBFM_SRC_FM_LEARN_SGD_ELEMENT_BPR_BLOCKS_COMPUTE_RANKED_LIST_PARALLEL_H_ */
