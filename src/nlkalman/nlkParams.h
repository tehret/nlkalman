/*
 * Copyright (c) 2019, Thibaud Ehret <ehret.thibaud@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef NLKPARAMS_H_INCLUDED
#define NLKPARAMS_H_INCLUDED

/**
 * @brief Structures of parameters dedicated to NL-Kalman
 *
 * @param sigma: Noise std;
 * @param sizePatch: size of patches (sizePatch x sizePatch);
 * @param nSimilarPatches: number of similar patches wanted;
 * @param sizeSearchWindow: size of the search window around the reference patch;
 * @param offSet: step between two similar patches;
 * @param rank: assumed maximum rank of covariance matrices;
 * @param beta: parameter used to estimate the covariance matrix;
 * @param verbose: if true, print informations;
 * @param a: memory parameter of the system;
 * @param occ: occlusion threshold;
 * @param flatAreaTrick: if true, use the homogeneous area trick;
 * @param noInnovTrick: if true, use the no innovationtrick;
 **/
struct nlkParams
{
	float sigma;               
	unsigned sizePatch;        
	unsigned channels;         
	unsigned nSimilarPatches; 
	unsigned sizeSearchWindow;
	unsigned offSet;          
	unsigned rank;            
	float beta;                
	bool verbose;

	float a;
	float occ;
	bool flatAreaTrick;
	bool noInnovTrick;
	float gammaR;
	float gammaT;

};

#endif
