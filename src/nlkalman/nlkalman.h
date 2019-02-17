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

#ifndef NLKALMAN_H_INCLUDED
#define NLKALMAN_H_INCLUDED

#include "nlkParams.h"
#include "LibVideoT.hpp"
#include "Utilities.h"
#include "LibMatrix.h"
#include "parametric_utils.h"
#include "parametric_transformation.h"

#include <stdexcept>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <ctime>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#define NUM_THREADS 8

/**
 * @brief Comparaison function used to sort an array made of pair by only considering the first component for the sort process
 * @param a: first element to be compared
 * @param b: second element to be compared
 **/
bool compFirst(std::pair<float, unsigned> a, std::pair<float, unsigned> b) 
{ 
	return (a.first > b.first); 
}

/**
 * @brief Structure containing matrices used in the Bayesian estimation (preallocated memory).
 **/
struct matWorkspace
{
	std::vector<float> groupTranspose;
	std::vector<float> baricenter;
	std::vector<float> covMat;
	std::vector<float> covEigVecs;
	std::vector<float> covEigVals;
	std::vector<float> eigVecsBackup;
	std::vector<float> inno;
	std::vector<float> F;
	std::vector<float> update;
};

/**
 * @brief Structure representing the Kalman filter.
 *
 **/
struct GG
{
	// coords of the patches used for the group at time k-1
	std::vector<std::pair<float, float> > coords;
	// Covariance matrix for the kalman filter
	std::vector<float> CF;
	// Covariance of predicted state
	std::vector<float> P;	
	// output patches to be aggregated at the end
	std::vector<float> patches;

    // If set to true the group is removed at the end of the kalman update
	bool set_destruction;
};

/**
 * @brief Initialize Parameters of NL-Kalman.
 *
 * @param o_params   : will contain the nlkParams;
 * @param p_sigma    : standard deviation of the noise;
 * @param p_size     : size of the video;
 * @param p_verbose  : if true, print some informations.
 * @param flatAreaTrick: if true, use the homogeneous area trick;
 * @param noInnovTrick : if true, use the no innovation trick;
 * @param a : memory parameter of the system.
 * @param occ : threshold for detecting occlusion.
 * @param rank : assumed rank of the covariance matrices.
 *
 * @return none.
 **/
void initializeNlkParameters(
	nlkParams &o_params
,	const float p_sigma
,	const VideoSize &p_size
,	const bool p_verbose
,	const bool flatAreaTrick
,	const bool noInnovTrick
,	const float a = 0.9
,	const float occ = 8.25
,	const unsigned rank = 4
);

/**
 * @brief Update search window.
 *
 * @param prms            : params to be updated;
 * @param sizeSearchWindow: size of the search window;
 *
 * @return none.
 **/
void setSizeSearchWindow(nlkParams& prms, unsigned sizeSearchWindow);

/**
 * @brief Update size patch.
 *
 * @param prms     : params to be updated;
 * @param size     : size of the video;
 * @param sizePatch: size of the patches;
 *
 * @return none.
 **/
void setSizePatch(nlkParams& prms, const VideoSize &size, unsigned sizePatch);

/**
 * @brief Update number of similar patches.
 *
 * @param prms           : params to be updated;
 * @param nSimilarPatches: number of similar patches;
 *
 * @return none.
 **/
void setNSimilarPatches(nlkParams& prms, unsigned nSimilarPatches);

/**
 * @brief Display parameters of the NL-Kalman.
 *
 * @param i_params : nlbParams for first or second step of the algorithm;
 *
 * @return none.
 **/
void printNlkParameters(
	const nlkParams &i_params
);

/**
 * @brief Main function the algorithm.
 *
 * @param i_imNoisy : contains the noisy video;
 * @param o_imFinal : will contain the final denoised image;
 * @param o_subFinal: will contain the final denoised image (subpixelic estimate);
 * @param of: optical flow;
 * @param H : parametric transform matrices;
 * @param nparams : number of parameters of H;
 * @param p_params: parameters for NL-Bayes.
 *
 * @return none.
 **/
void nlKalman(
	Video<float> &i_imNoisy
,	Video<float> &o_imFinal
,	Video<float> &o_subFinal
,	Video<float> &of
,	float *H
,	int nparams
,	const nlkParams p_params
);

/**
 * @brief Estimate the best similar patches to a reference one.
 *
 * @param i_im: contains the noisy video on which distances are processed;
 * @param current: group in which these patches will be loaded;
 * @param pidx: spatial location of the reference patch;
 * @param frame: temporal location of the reference patch.
 * @param nlkParams: parameters.
 *
 * @return number of similar patches found.
 **/
unsigned estimateSimilarPatches(
	Video<float> const& i_im
,	GG* current
,	const std::pair<int,int> pidx
,	const int frame
,	const nlkParams &p_params
);

/**
 * @brief Compute the Bayes estimation assuming a low rank covariance matrix.
 *
 * @param current: contains all similar patches. Will contain estimates for all similar patches;
 * @param i_mat: contains allocated memory for processing (See matWorkspace for more explanation);
 * @param p_params: contains parameters (See nlkParams for more explanation).
 *
 * @return none.
 **/
void computeBayesEstimate(
	GG* current
,	matWorkspace &i_mat
,	nlkParams const& p_params
);

/**
 * @brief Compute the Kalman estimation.
 *
 * @param i_imNoisy: contains the noisy video;
 * @param i_imFinal: will contain the final video;
 * @param of: contains the optical flow;
 * @param H : parametric transform matrices;
 * @param nparams: number of parameters of H;
 * @param frame: contains the index of the current frame;
 * @param current: contains all similar patches. Will contain estimates for all similar patches;
 * @param i_mat: contains allocated memory for processing (See matWorkspace for more explanation);
 * @param p_params: contains parameters (See nlkParams for more explanation).
 *
 * @return none.
 **/
void computeKalmanEstimate(
	Video<float> &i_imNoisy
,	Video<float> &o_imFinal
,	Video<float> &of
,	float* H
,	int nparams
,	const int frame
,	GG* current
,	matWorkspace &i_mat
,	nlkParams const& p_params
);


/**
 * @brief Aggregate estimates of all similar patches contained in a group.
 *
 * @param weights: subpixelic weights to be updated;
 * @param x, y: spatial location being updated;
 * @param f: index of the frame
 * @param precomputations: contains precomputation used for the subpixelic weights;
 *
 **/
void updateWeights(
	Video<float>& weights
,	float x
,	float y
,	int f
,	const std::vector<float>& precomputations
);

/**
 * @brief Aggregate estimates of all similar patches contained in a group.
 *
 * @param io_im: update the image with estimate values;
 * @param weightsSub: contains the aggregation values for the subpixelic processing;
 * @param groups: groups which need to be aggregated;
 * @param frame: index of the frame currently being processed;
 * @param p_params: parameters;
 * @param precomputations: contains precomputation used for the subpixelic weights;
 *
 **/
void computeAggregation(
	Video<float> &io_im
,	Video<float> &weightsSub
,	std::vector<GG*> &groups
, 	int frame
,	const nlkParams& p_params
, 	const std::vector<float>& precomputations
);

/**
 * @brief Compute the final weighted aggregation.
 *
 * @param i_im: will contain the final video;
 * @param i_weight: associated weight for each estimate of pixels.
 * @param frame: index of the frame on which to do aggregation
 *
 * @return none.
 **/
void computeWeightedAggregation(
	Video<float> &i_im
,	const Video<float> &i_weight
,	int frame
);

/**
 * @brief Compute the final weighted aggregation (subpixelic version).
 *
 * @param i_im: will contain the final video;
 * @param i_weight: associated weight for each estimate of pixels.
 * @param frame: index of the frame on which to do aggregation
 *
 * @return none.
 **/
void computeWeightedAggregationSub(
	Video<float> &i_im
,	const Video<float> &i_weight
,	int frame
);

#endif // NLKALMAN_H_INCLUDED
