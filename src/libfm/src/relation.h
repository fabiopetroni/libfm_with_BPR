// Copyright (C) 2011, 2012, 2013, 2014 Steffen Rendle
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
// relation.h: Data and Links for Relations

#ifndef RELATION_DATA_H_
#define RELATION_DATA_H_

#include "Data.h"
#include <limits>
#include "../../util/matrix.h"
#include "../../util/fmatrix.h"
#include "../../fm_core/fm_data.h"
#include "../../fm_core/fm_model.h"

class RelationData {
	protected:
		uint cache_size;
		bool has_xt;
		bool has_x;
	public:
		RelationData(uint cache_size, bool has_x, bool has_xt) {
			this->data_t = NULL;
			this->data = NULL;
			this->cache_size = cache_size;
			this->has_x = has_x;
			this->has_xt = has_xt;
			this->meta = NULL;
		}
		DataMetaInfo* meta;

		LargeSparseMatrix<DATA_FLOAT>* data_t;
		LargeSparseMatrix<DATA_FLOAT>* data;

		int num_feature;
		uint num_cases;
		uint attr_offset;

		void load(std::string filename);
		void debug();
};


class RelationJoin {
	public:
		DVector<uint> data_row_to_relation_row;
		RelationData* data;

		void load(std::string filename, uint expected_row_count) {
			bool do_binary = false;
			// check if binary or text format should be read
			{
				std::ifstream in (filename.c_str(), std::ios_base::in | std::ios_base::binary);
				if (in.is_open()) {
					uint file_version;
					uint data_size;
					in.read(reinterpret_cast<char*>(&file_version), sizeof(file_version));
					in.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
					do_binary = ((file_version == DVECTOR_EXPECTED_FILE_ID) && (data_size == sizeof(uint)));
					in.close();
				}
			}
			if (do_binary) {
				//std::cout << "(binary mode) " << std::endl;
				data_row_to_relation_row.loadFromBinaryFile(filename);
			} else {
				//std::cout << "(text mode) " << std::endl;
				data_row_to_relation_row.setSize(expected_row_count);
				data_row_to_relation_row.load(filename);
			}
			assert(data_row_to_relation_row.dim == expected_row_count);
		}

		//VALIDATION added by Fabio
		DVector<uint> validation_data_row_to_relation_row;

		void loadValidation(std::string filename_validation, uint expected_row_count_validation){
			bool do_binary = false;
			// check if binary or text format should be read
			{
				std::ifstream in (filename_validation.c_str(), std::ios_base::in | std::ios_base::binary);
				if (in.is_open()) {
					uint file_version;
					uint data_size;
					in.read(reinterpret_cast<char*>(&file_version), sizeof(file_version));
					in.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
					do_binary = ((file_version == DVECTOR_EXPECTED_FILE_ID) && (data_size == sizeof(uint)));
					in.close();
				}
			}
			if (do_binary) {
				//std::cout << "(binary mode) " << std::endl;
				validation_data_row_to_relation_row.loadFromBinaryFile(filename_validation);
			} else {
				//std::cout << "(text mode) " << std::endl;
				validation_data_row_to_relation_row.setSize(expected_row_count_validation);
				validation_data_row_to_relation_row.load(filename_validation);
			}
			assert(validation_data_row_to_relation_row.dim == expected_row_count_validation);
		}
};

void RelationData::load(std::string filename) {

	std::cout << "has x = " << has_x << std::endl;
	std::cout << "has xt = " << has_xt << std::endl;
	assert(has_x || has_xt);

	try{	//try to load matrices in binary format
		//uint num_cases = 0;
		uint num_values = 0;
		uint this_cs = cache_size;
		if (has_xt && has_x) { this_cs /= 2; }

		if (has_x) {
			std::cout << "data... ";
			this->data = new LargeSparseMatrixHD<DATA_FLOAT>(filename + ".x", this_cs);
			this->num_feature = this->data->getNumCols();
			num_values = this->data->getNumValues();
			num_cases = this->data->getNumRows();
		} else {
			data = NULL;
		}
		if (has_xt) {
			std::cout << "data transpose... ";
			this->data_t = new LargeSparseMatrixHD<DATA_FLOAT>(filename + ".xt", this_cs);
			this->num_feature = this->data_t->getNumRows();
			num_values = this->data_t->getNumValues();
			num_cases = this->data_t->getNumCols();
		} else {
			data_t = NULL;
		}

		if (has_xt && has_x) {
			assert(this->data->getNumCols() == this->data_t->getNumRows());
			assert(this->data->getNumRows() == this->data_t->getNumCols());
			assert(this->data->getNumValues() == this->data_t->getNumValues());
		}

	}
	catch (int e){
		//MATRICES ARE IN TEXT FORMAT
		std::string text_format_filename = filename + ".x";
		this->data = new LargeSparseMatrixMemory<DATA_FLOAT>();
		DVector< sparse_row<DATA_FLOAT> >& data = ((LargeSparseMatrixMemory<DATA_FLOAT>*)this->data)->data;

		int num_rows = 0;
		uint64 num_values = 0;
		num_feature = 0;
		bool has_feature = false;

		// (1) determine the number of rows and the maximum feature_id
		{
			std::ifstream fData(text_format_filename.c_str());
			if (! fData.is_open()) {
				throw "unable to open " + text_format_filename;
			}
			DATA_FLOAT _value;
			int nchar, _feature;
			while (!fData.eof()) {
				std::string line;
				std::getline(fData, line);

//				std::cout << "line=" << line << std::endl;

				const char *pline = line.c_str();
				while ((*pline == ' ')  || (*pline == 9)) { pline++; } // skip leading spaces
				if ((*pline == 0)  || (*pline == '#')) { continue; }  // skip empty rows
//				if (sscanf(pline, "%f%n", &_value, &nchar) >=1) {
//					pline += nchar;
//					min_target = std::min(_value, min_target);
//					max_target = std::max(_value, max_target);
					num_rows++;
					while (sscanf(pline, "%d:%f%n", &_feature, &_value, &nchar) >= 2) {
						pline += nchar;
						num_feature = std::max(_feature, num_feature);
						has_feature = true;
						num_values++;
					}
					while ((*pline != 0) && ((*pline == ' ')  || (*pline == 9))) { pline++; } // skip trailing spaces
					if ((*pline != 0)  && (*pline != '#')) {
						throw "cannot parse line \"" + line + "\" at character " + pline[0];
					}
//				} else {
//					throw "cannot parse line \"" + line + "\" at character " + pline[0];
//				}
			}
			fData.close();
		}

		if (has_feature) {
			num_feature++; // number of feature is bigger (by one) than the largest value
		}

		num_cases = num_rows;
		std::cout << "num_cases=" << this->num_cases << "\tnum_values=" << num_values << "\tnum_features=" << this->num_feature << std::endl;

		data.setSize(num_rows);
//		target.setSize(num_rows);

		((LargeSparseMatrixMemory<DATA_FLOAT>*)this->data)->num_cols = num_feature;
		((LargeSparseMatrixMemory<DATA_FLOAT>*)this->data)->num_values = num_values;

		MemoryLog::getInstance().logNew("data_float", sizeof(sparse_entry<DATA_FLOAT>), num_values);
		sparse_entry<DATA_FLOAT>* cache = new sparse_entry<DATA_FLOAT>[num_values];

		// (2) read the data
		{
			std::ifstream fData(text_format_filename.c_str());
			if (! fData.is_open()) {
				throw "unable to open " + text_format_filename;
			}
			int row_id = 0;
			uint64 cache_id = 0;
			DATA_FLOAT _value;
			int nchar, _feature;
			while (!fData.eof()) {
				std::string line;
				std::getline(fData, line);
				const char *pline = line.c_str();
				while ((*pline == ' ')  || (*pline == 9)) { pline++; } // skip leading spaces
				if ((*pline == 0)  || (*pline == '#')) { continue; }  // skip empty rows
//				if (sscanf(pline, "%f%n", &_value, &nchar) >=1) {
//					pline += nchar;
					assert(row_id < num_rows);
//					target.value[row_id] = _value;
					data.value[row_id].data = &(cache[cache_id]);
					data.value[row_id].size = 0;

					while (sscanf(pline, "%d:%f%n", &_feature, &_value, &nchar) >= 2) {
						pline += nchar;
						assert(cache_id < num_values);
						cache[cache_id].id = _feature;
						cache[cache_id].value = _value;
						cache_id++;
						data.value[row_id].size++;
					}
					row_id++;

					while ((*pline != 0) && ((*pline == ' ')  || (*pline == 9))) { pline++; } // skip trailing spaces
					if ((*pline != 0)  && (*pline != '#')) {
						throw "cannot parse line \"" + line + "\" at character " + pline[0];
					}
//				} else {
//					throw "cannot parse line \"" + line + "\" at character " + pline[0];
//				}
			}
			fData.close();

			assert(num_rows == row_id);
			assert(num_values == cache_id);
		}
	}

	meta = new DataMetaInfo(this->num_feature);

	if (fileexists(filename + ".groups")) {
		meta->loadGroupsFromFile(filename + ".groups");
	}
}


void RelationData::debug() {
	if (has_x) {
		for (data->begin(); (!data->end()) && (data->getRowIndex() < 4); data->next() ) {
			for (uint j = 0; j < data->getRow().size; j++) {
				std::cout << " " << data->getRow().data[j].id << ":" << data->getRow().data[j].value;
			}
			std::cout << std::endl;
		}
	}
}

#endif /*RELATION_DATA_H_*/
