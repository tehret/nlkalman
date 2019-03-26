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

/**
 * @file nlkalman.cpp
 * @brief NL-Kalman denoising functions
 *
 * @author Thibaud Ehret <ehret.thibaud@gmail.com>
 **/


#include "nlkalman.h"

// Parameters for the subpixelic estimation, only change if you know what you're doing
#define ORDER 2
#define NPRECOMP 1000000 
#define R2MAX 32
#define SIGMA2 0.25
#define STEP (NPRECOMP/R2MAX)
// RADIUS must be smaller than R2MAX
#define RADIUS2 (16*SIGMA2)
#define INDMAX (std::sqrt(RADIUS2))

#define BICUBIC
#define EIGTHRESH 2

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
,	const float a
,	const float occ
,	const unsigned rank
){
	//! Standard deviation of the noise
	o_params.sigma = p_sigma;
    o_params.sizePatch = 8;
	o_params.channels = p_size.channels;
    o_params.nSimilarPatches = 64;

	//! Step between initial patches
	o_params.offSet     = std::max((unsigned)1, o_params.sizePatch     / 2);
	o_params.sizeSearchWindow = 15;
    o_params.beta = 1.f;

	// maximum rank of covariance matrix
	o_params.rank = rank;

	//! Print information?
	o_params.verbose = p_verbose;

	o_params.a = a;
	o_params.occ = occ;
	o_params.flatAreaTrick = flatAreaTrick;
	o_params.noInnovTrick = noInnovTrick;

    // Used of the different tricks
	o_params.gammaR = 1.05f;
	o_params.gammaT = 1.2f;
}

/**
 * @brief Update search window.
 *
 * @param prms            : params to be updated;
 * @param sizeSearchWindow: size of the search window;
 *
 * @return none.
 **/
void setSizeSearchWindow(nlkParams& prms, unsigned sizeSearchWindow)
{
	prms.sizeSearchWindow = sizeSearchWindow;
}

/**
 * @brief Update size patch.
 *
 * @param prms     : params to be updated;
 * @param size     : size of the video;
 * @param sizePatch: size of the patches;
 *
 * @return none.
 **/
void setSizePatch(nlkParams& prms, const VideoSize &size, unsigned sizePatch)
{
	prms.sizePatch = sizePatch;
	prms.offSet = std::max(1, (int)sizePatch/2);
}

/**
 * @brief Update number of similar patches.
 *
 * @param prms           : params to be updated;
 * @param nSimilarPatches: number of similar patches;
 *
 * @return none.
 **/
void setNSimilarPatches(nlkParams& prms, unsigned nSimilarPatches)
{
	prms.nSimilarPatches = nSimilarPatches;
	prms.nSimilarPatches = std::min(nSimilarPatches, prms.sizeSearchWindow *
	                                                 prms.sizeSearchWindow);
}

/**
 * @brief Display parameters of NL-Kalman.
 *
 * @param i_params : parameters of the method;
 *
 * @return none.
 **/
void printNlkParameters(
	const nlkParams &i_prms
){
	printf("Parameters\n");
	printf("\tNoise std                     = %d\n"       , i_prms.sigma);
	printf("\tPatch search:\n");
	printf("\t\tPatch size                  = %d\n"       , i_prms.sizePatch);
	printf("\t\tNumber of patches           = %d\n"       , i_prms.nSimilarPatches);
	printf("\t\tSpatial search window       = %dx%d\n"    , i_prms.sizeSearchWindow, i_prms.sizeSearchWindow);
	printf("\tGroup filtering:\n");
	printf("\t\tBeta                        = %g\n"       , i_prms.beta);
	printf("\t\tRank                        = %d\n"       , i_prms.rank);
	printf("\t\tAlpha                       = %f\n"       , i_prms.a);
	printf("\t\tOcclusion                   = %f\n"       , i_prms.occ);
	printf("\tSpeed-ups:\n");
	printf("\t\tOffset                      = %d\n"       , i_prms.offSet);
	printf("\t\tFlat Trick                  = %d\n"       , i_prms.flatAreaTrick);
	printf("\t\tInnov Trick                 = %d\n"       , i_prms.noInnovTrick);
}


#ifndef BICUBIC
/**
 * @brief Compute the bilinear interpolation of the requested position.
 *
 * @param v          : reference video;
 * @param x, y, f, ch: location of the requested pixel in the video (x and y are floating 
 *                     values while f and ch are integers);
 *
 * @return the interpolated value at the requested location.
 **/
inline float interpolate(const Video<float> &v, float x, float y, int f, int ch)
{
	int x1 = std::floor(x), x2 = x1 + 1;
	int y1 = std::floor(y), y2 = y1 + 1;

	float f1 = (y2-y) * v(x1, y1, f, ch) + (y-y1) * v(x1, y2, f, ch);
	float f2 = (y2-y) * v(x2, y1, f, ch) + (y-y1) * v(x2, y2, f, ch);

	return (x2-x) * f1 + (x-x1) * f2;	
}
#else
// Cubic kernel
inline static
float cubic_interpolation(float v[4], float x)
{
	return v[1] + 0.5 * x*(v[2] - v[0]
			+ x*(2.0*v[0] - 5.0*v[1] + 4.0*v[2] - v[3]
			+ x*(3.0*(v[1] - v[2]) + v[3] - v[0])));
}

//bicubic interpolation
static float bicubic_interpolation(float p[4][4], float x, float y)
{
	float v[4];

	v[0] = cubic_interpolation(p[0], y);
	v[1] = cubic_interpolation(p[1], y);
	v[2] = cubic_interpolation(p[2], y);
	v[3] = cubic_interpolation(p[3], y);
	return cubic_interpolation(v, x);
}

/**
 * @brief Compute the bicubic interpolation of the requested position.
 *
 * @param v          : reference video;
 * @param x, y, f, ch: location of the requested pixel in the video (x and y are floating 
 *                     values while f and ch are integers);
 *
 * @return the interpolated value at the requested location.
 **/
inline float interpolate(const Video<float>& v, float x, float y, int f, int ch)
{
	x -= 1;
	y -= 1;

	int ix = floor(x);
	int iy = floor(y);

	float c[4][4];
	for (int j = 0; j < 4; j++)
		for (int i = 0; i < 4; i++)
			if(ix+i < 0 || ix+i >= v.sz.width || iy+j < 0 || iy+j >= v.sz.height)
				c[i][j] = 0;
			else
				c[i][j] = v(ix+i,iy+j,f,ch);

	return bicubic_interpolation(c, x - ix, y - iy);
}
#endif


/**
 * @brief Precompute values for the subpixelic computation.
 *
 * @param prec          : will contains the precomputed values;
 *
 * @return none.
 **/
inline void precomputeNormalizedConvolutionCoefs(std::vector<float>& prec)
{
	float sigma3 = R2MAX/(2*SIGMA2*prec.size());
	for(int i = 0; i < prec.size(); ++i)
	       prec[i] = std::exp(-i*sigma3);	
}

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
,	const nlkParams p_prms
){
	const unsigned sPx  = p_prms.sizePatch;
	const unsigned sPt  = 1;
	const unsigned sPC  = p_prms.sizePatch * p_prms.sizePatch
	                    * p_prms.channels;
	const unsigned r    = p_prms.rank;
	const unsigned p_nSimP = p_prms.nSimilarPatches;

	unsigned nThreads = 1;
#ifdef _OPENMP
	nThreads = omp_get_max_threads();
	printf("Number of threads available: %d, number wanted: %d\n", nThreads, NUM_THREADS);
	nThreads = std::min(nThreads, (unsigned)NUM_THREADS);
#endif
	printf("Using %d threads\n", nThreads);

    // Allocate memory
	o_imFinal.resize(i_imNoisy.sz);
	o_subFinal.resize(i_imNoisy.sz);

	// Liste of groups (pointers so that we can delete them and move them for cheap
	std::vector<GG*> groups;

	// Aggregation weights for subpixelic version
#if ORDER == 2
	Video<float> weightsSub(i_imNoisy.sz.width, i_imNoisy.sz.height, i_imNoisy.sz.frames, 15+6*p_prms.channels);
#elif ORDER == 1
	Video<float> weightsSub(i_imNoisy.sz.width, i_imNoisy.sz.height, i_imNoisy.sz.frames, 6+3*p_prms.channels);
#else
	Video<float> weightsSub(i_imNoisy.sz.width, i_imNoisy.sz.height, i_imNoisy.sz.frames, 1+p_prms.channels);
#endif
	// Aggregation weights
	Video<float> weights(i_imNoisy.sz.width, i_imNoisy.sz.height, i_imNoisy.sz.frames);

	matWorkspace workspace; 
	//allocate workspace
	workspace.baricenter.resize(sPC);
	workspace.covMat.resize(sPC*sPC);
	workspace.covEigVecs.resize(sPC*r);
	workspace.covEigVals.resize(sPC*r);
	workspace.groupTranspose.resize(2*p_nSimP*r);
	workspace.inno.resize(2*p_nSimP*sPC);	
	workspace.F.resize(2*p_nSimP*sPC);	
	workspace.update.resize(sPC*sPC);	

	std::vector<float> precomputations(NPRECOMP);
	precomputeNormalizedConvolutionCoefs(precomputations);


	// For each frame
	for(int f = 0; f < i_imNoisy.sz.frames; ++f)
	{
        if(p_prms.verbose)
            printf("Processing frame %d\n", f);

#ifdef DEBUG
		printf("Exporting groups\n");
		std::string file_path = "groups_" + std::to_string(f) + ".txt";
		FILE* file = fopen(file_path.c_str(), "w");
		for(int g = 0; g < groups.size(); ++g)
		{
			for(int idx = 0; idx < groups[g]->coords.size(); ++idx)
				fprintf(file, "%f %f ", groups[g]->coords[idx].first, groups[g]->coords[idx].second);
			fprintf(file, "\n");
		}
		fclose(file);
#endif


		// First part is Kalman denoising:
		// The groups are propagate from the previous frame to the current frame 
#ifdef _OPENMP
#pragma omp parallel for num_threads(nThreads) schedule(dynamic) \
		shared(groups) \
		firstprivate(p_prms, workspace)
#endif
		for(int g = 0; g < groups.size(); ++g)
			computeKalmanEstimate(i_imNoisy, o_imFinal, of, H, nparams, f, groups[g], workspace, p_prms);

		for(int g = 0; g < groups.size(); ++g)
		{
			// update both subpixelic weights and regular weights
			for(int idx = 0; idx < groups[g]->coords.size(); ++idx)
			{
				float newx = groups[g]->coords[idx].first;
				float newy = groups[g]->coords[idx].second;

				for (int hy = 0; hy < sPx; hy++)
				for (int hx = 0; hx < sPx; hx++)
				{
					updateWeights(weightsSub, newx + hx, newy + hy, f, precomputations);
					weights(std::round(newx)+hx, std::round(newy)+hy, f)++;
				}
			}
		}
		
		// delete useless groups (groups that don't contain patches anymore)
		for(int g = groups.size() - 1; g >= 0; --g)
		{
			if(groups[g]->set_destruction)
			{
				if(g == groups.size()-1)
				{
					delete groups[g];
					groups.pop_back();
				}
				else
				{
					delete groups[g];
					groups[g] = groups[groups.size() - 1];
					groups.pop_back();
				}
			}
		}

		// Do a classic denoising for the rest:
		// Find the indexes of the patch that still needs denoising + initialize the new groups
		std::vector<GG*> newGroups;

		// Compute the different regions needed;
		std::vector<std::pair<int,int> > regionX{{0,i_imNoisy.sz.width-sPx},{i_imNoisy.sz.width-sPx, i_imNoisy.sz.width-sPx+1}, {0, i_imNoisy.sz.width-sPx}, {i_imNoisy.sz.width-sPx, i_imNoisy.sz.width-sPx+1}};
		std::vector<std::pair<int,int> > regionY{{0, i_imNoisy.sz.height-sPx},{0, i_imNoisy.sz.height-sPx}, {i_imNoisy.sz.height-sPx, i_imNoisy.sz.height-sPx+1}, {i_imNoisy.sz.height-sPx, i_imNoisy.sz.height-sPx+1}};
		for(int region = 0; region < regionX.size(); ++region)
		for(int x = regionX[region].first; x < regionX[region].second; x += p_prms.offSet)
		for(int y = regionY[region].first; y < regionY[region].second; y += p_prms.offSet)
		{
			bool compute = false;
            // Check if the requested patch has already been computed
			for (int hy = 0; hy < sPx; hy++)
			for (int hx = 0; hx < sPx; hx++)
			{
				if(weights(x + hx, y + hy, f, 0) <= 1)
					compute = true;
			}

			if(compute)
			{
				GG* current = new GG();

				estimateSimilarPatches(i_imNoisy, current, std::make_pair(x,y), f, p_prms);
                // Add the new group to the list
                newGroups.push_back(current);
                groups.push_back(current);
                // Update the weights of the element considered
                for(int idx = 0; idx < current->coords.size(); ++idx)
                for (int hy = 0; hy < sPx; hy++)
                for (int hx = 0; hx < sPx; hx++)
                {
                    updateWeights(weightsSub, current->coords[idx].first + hx, current->coords[idx].second + hy, f, precomputations);
                    ++weights(std::round(current->coords[idx].first) + hx, std::round(current->coords[idx].second) + hy, f);
                }
			}
		}

		// Compute the denoising for the new groups
#ifdef _OPENMP
#pragma omp parallel for num_threads(nThreads) schedule(dynamic) \
		shared(newGroups) \
		firstprivate(p_prms, workspace)
#endif
		for(int g = 0; g < newGroups.size(); ++g)
			computeBayesEstimate(newGroups[g], workspace, p_prms);

		// Aggregate the data from the groups 
		computeAggregation(o_imFinal, weightsSub, groups, f, p_prms, precomputations);
		computeWeightedAggregationSub(o_subFinal, weightsSub, f);
		computeWeightedAggregation(o_imFinal, weights, f);

	}

	// Clean the groups now that we are done
	while(!groups.empty()) delete groups.back(), groups.pop_back();
}

/**
 * @brief Estimate the best similar patches to a reference one and load them into a group.
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
	Video<float> const& i_imNoisy
,	GG* current
,	const std::pair<int,int> pidx
,	const int frame
,	const nlkParams &p_params
){
	//! Initialization
	int sWx   = p_params.sizeSearchWindow;
	int sWy   = p_params.sizeSearchWindow;
	const int sPx   = p_params.sizePatch;
	const VideoSize sz = i_imNoisy.sz;
	const int chnls = sz.channels;

	//! Allocate vector of patch distances
	std::vector<std::pair<float, std::pair<int,int> > > distance(sWx * sWy);

	//! Number of patches in search region
	int nsrch = 0;

	//! Spatial search range
	int rangex[2];
	int rangey[2];

	rangex[0] = std::max(0, pidx.first  - (sWx-1)/2);
	rangey[0] = std::max(0, pidx.second - (sWy-1)/2);

	rangex[1] = std::min((int)sz.width  - sPx, pidx.first  + (sWx-1)/2);
	rangey[1] = std::min((int)sz.height - sPx, pidx.second + (sWy-1)/2);

	//! Compute distance between patches in search range
	for (int qy = rangey[0], dy = 0; qy <= rangey[1]; qy++, dy++)
		for (int qx = rangex[0], dx = 0; qx <= rangex[1]; qx++, dx++)
		{
			//! Squared L2 distance
			float dist = 0.f, dif;
			for (int c = 0; c < chnls; c++)
			for (int hy = 0; hy < sPx; hy++)
			for (int hx = 0; hx < sPx; hx++)
				dist += (dif = i_imNoisy(pidx.first + hx, pidx.second + hy, frame, c)
						- i_imNoisy(qx + hx, qy + hy, frame, c)) * dif;

			//! Save distance and corresponding patch index
			distance[nsrch++] = std::make_pair(dist, std::make_pair(qx, qy));
		}

	distance.resize(nsrch);

	//! Keep only the nSimilarPatches best similar patches
	unsigned nSimP = std::min(p_params.nSimilarPatches, (unsigned)distance.size());
	std::partial_sort(distance.begin(), distance.begin() + nSimP,
	                  distance.end(), comparaisonFirst);

	current->coords.resize(nSimP);
	//! Register position of similar patches
	for (unsigned n = 0; n < nSimP; n++)
		current->coords[n] = distance[n].second;

	current->patches.resize(chnls*sPx*sPx*2*nSimP);
	//! Save similar patches into 3D groups
	const unsigned w   = sz.width;
	const unsigned wh  = sz.wh;
	const unsigned whc = sz.whc;
	for (unsigned c = 0, k = 0; c < chnls; c++)
	for (unsigned hx = 0; hx < sPx; hx++)
	for (unsigned hy = 0; hy < sPx; hy++)
	for (unsigned n = 0; n < nSimP; n++, k++)
		current->patches[k] = i_imNoisy(std::round(current->coords[n].first) + hx, std::round(current->coords[n].second) + hy, frame, c); 

	return nSimP;
}

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
){
	//! Parameters initialization
	const float sigma2 = p_params.beta * p_params.sigma * p_params.sigma;
	const unsigned sPC  = p_params.sizePatch * p_params.sizePatch
	                    * p_params.channels;
	const unsigned r    = std::min(p_params.rank, sPC);

	const unsigned p_nSimP = current->coords.size();


    // Trick that detects if a patch is flat and then denoise it accordingly, first introduced in Marc Lebrun's NL-Bayes algorithm (Marc Lebrun, Antoni Buades, and Jean-Michel Morel, Implementation of the "Non-Local Bayes" (NL-Bayes) Image Denoising Algorithm, Image Processing On Line, 3 (2013), pp. 1–42. https://doi.org/10.5201/ipol.2013.16)
	if(p_params.flatAreaTrick)
	{
		//! Parameters
		const float threshold = sigma2 * p_params.gammaR;
		const unsigned sP2  = p_params.sizePatch * p_params.sizePatch;

		//! Compute the standard deviation of the set of patches
		const float stdDev = computeStdDeviation(current->patches, sP2, p_nSimP, p_params.channels);

		//! If we are in an homogeneous area
		if (stdDev < threshold) {
			// The denoising is compute simply by averaging everything
			for (unsigned c = 0; c < p_params.channels; c++) {
				float mean = 0.f;

				for (unsigned n = 0; n < p_nSimP; n++) {
					for (unsigned k = 0; k < sP2; k++) {
						mean += current->patches[n * sPC + c * sP2 + k];
					}
				}

				mean /= float(sP2 * p_nSimP);

				for (unsigned n = 0; n < p_nSimP; n++) {
					for (unsigned k = 0; k < sP2; k++) {
						current->patches[n * sPC + c * sP2 + k] = mean;
					}
				}
			}
			// Initialize the Kalman filter
			current->CF.resize(sPC*sPC);
			for(int i = 0; i < sPC*sPC; ++i)
				current->CF[i] = 0;
			current->P = current->CF;
			return;
		}
	}

	//! Center 3D groups around their baricenter
	centerData(current->patches, i_mat.baricenter, p_nSimP, sPC);

	float r_variance = 0.f;
	float total_variance = 1.f;

	if (r > 0)
	{
		//! Compute the covariance matrix of the set of similar patches
		covarianceMatrix(current->patches, i_mat.covMat, p_nSimP, sPC);


		//! Compute leading eigenvectors
		int info = matrixEigs(i_mat.covMat, sPC, r, i_mat.covEigVals, i_mat.covEigVecs);

		//! Compute eigenvalues-based coefficients of Bayes' filter
		for (unsigned i = 0; i < r; ++i)
			i_mat.covEigVals[i] -= std::min(i_mat.covEigVals[i], sigma2);

		// Compute CF here before "damaging" covEigVals
		i_mat.eigVecsBackup = i_mat.covEigVecs;
		float *eigVecs = i_mat.eigVecsBackup.data();
		for (unsigned k = 0; k < r  ; ++k)
		for (unsigned i = 0; i < sPC; ++i)
			*eigVecs++ *= i_mat.covEigVals[k];

		current->CF.resize(sPC*sPC);
		productMatrix(current->CF,
					  i_mat.eigVecsBackup,
					  i_mat.covEigVecs,
					  sPC, sPC, r,
					  false, true);

		for (unsigned k = 0; k < r; ++k)
			i_mat.covEigVals[k] = (i_mat.covEigVals[k] < 1e-8f) ? 0. :1.f / ( 1. + sigma2 / i_mat.covEigVals[k] );


		current->P = current->CF;

		/* NOTE: io_groupNoisy, if read as a column-major matrix, contains in each
		 * row a patch. Thus, in column-major storage it corresponds to X^T, where
		 * each column of X contains a centered data point.
		 *
		 * We need to compute the noiseless estimage hX as 
		 * hX = U * W * U' * X
		 * where U is the matrix with the eigenvectors and W is a diagonal matrix
		 * with the filter coefficients.
		 *
		 * Matrix U is stored (column-major) in i_mat.covEigVecs. Since we have X^T
		 * we compute 
		 * hX' = X' * U * (W * U')
		 */

		//! Z' = X'*U
		productMatrix(i_mat.groupTranspose,
						  current->patches,
						  i_mat.covEigVecs,
						  p_nSimP, r, sPC,
						  false, false);

		//! U * W
		eigVecs = i_mat.covEigVecs.data();
		for (unsigned k = 0; k < r  ; ++k)
		for (unsigned i = 0; i < sPC; ++i)
			*eigVecs++ *= i_mat.covEigVals[k];

		//! hX' = Z'*(U*W)'
		productMatrix(current->patches,
						  i_mat.groupTranspose,
						  i_mat.covEigVecs,
						  p_nSimP, sPC, r,
						  false, true);

		//! Add baricenter
		for (unsigned j = 0, k = 0; j < sPC; j++)
			for (unsigned i = 0; i < p_nSimP; i++, k++)
				current->patches[k] += i_mat.baricenter[j];
	}
	else
	{
		for (unsigned j = 0, k = 0; j < sPC; j++)
			for (unsigned i = 0; i < p_nSimP; i++, k++)
				current->patches[k] = i_mat.baricenter[j];
	}
}


/**
 * @brief Implementation of computeKalmanEstimate computing the
 * principal directions of the a priori covariance matrix. This functions
 * computes the eigenvectors/values of the data covariance matrix using LAPACK.
 *
 * See computeBayesEstimate for information about the arguments.
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
){
	//! Parameters initialization
	float sigma2 = p_params.sigma * p_params.sigma;
	const unsigned sPC  = p_params.sizePatch * p_params.sizePatch
	                    * p_params.channels;
	const unsigned dim = p_params.sizePatch * p_params.sizePatch;
	const unsigned r    = std::min(sPC, p_params.rank);
	int sp = p_params.sizePatch;


	std::vector<bool> tytr(current->coords.size());

	for(int idx = current->coords.size() - 1; idx >= 0; --idx)
	{
		float oldx = current->coords[idx].first;
		float oldy = current->coords[idx].second;

		float newx, newy;
		float globalVarDiff = 0.;
		bool globalGood = false;
		// Try parametric registration first (if available), use at your own risk
		if(H != NULL)
		{
			project(oldx, oldy, &(H[(frame-1)*nparams]), newx, newy, nparams);

			if((newx >= 0) && (newy >= 0) && (newx <= i_imNoisy.sz.width - sp) && (newy <= i_imNoisy.sz.height - sp))
			{
				float dif;
				for(int c = 0; c < i_imNoisy.sz.channels; ++c)
				for(int dy = 0; dy < sp; ++dy)
				for(int dx = 0; dx < sp; ++dx)
					globalVarDiff += (dif = (i_imNoisy(std::round(newx) + dx, std::round(newy) + dy, frame, c) - o_imFinal(std::round(oldx) + dx, std::round(oldy) + dy, frame-1, c)))*dif;

				globalGood = true;

				// check if trajectory is not occluded and then use this one
				if(globalVarDiff < p_params.occ*sigma2*sPC)
				{
					tytr[idx] = true;
					continue;
				}
			}
		}

		// We use the center of the patch to get the optical flow (instead of using the one of the top left pixel)
		newx = oldx + of(std::round(oldx + (float)sp/2.), std::round(oldy + (float)sp/2.), frame-1, 0);
		newy = oldy + of(std::round(oldx + (float)sp/2.), std::round(oldy + (float)sp/2.), frame-1, 1);

		bool ofGood = true;
		if((newx < 0) || (newy < 0) || (newx > i_imNoisy.sz.width - sp) || (newy > i_imNoisy.sz.height - sp))
			ofGood = false;

		float ofVarDiff = 0.f;
		float dif;
		if(ofGood)
		{
			for(int c = 0; c < i_imNoisy.sz.channels; ++c)
			for(int dy = 0; dy < sp; ++dy)
			for(int dx = 0; dx < sp; ++dx)
				ofVarDiff += (dif = (i_imNoisy(std::round(newx) + dx, std::round(newy) + dy, frame, c) - o_imFinal(std::round(oldx) + dx, std::round(oldy) + dy, frame-1, c)))*dif;
		}

		float bestVarDiff = ofVarDiff;
		// Keep the best matching one
		if(globalGood && (!ofGood || globalVarDiff < ofVarDiff))
		{
			tytr[idx] = true;
			bestVarDiff = globalVarDiff;
		}
		else 
		{
			tytr[idx] = false;
			bestVarDiff = ofVarDiff;
		}

		// check if the best trajectory is not occluded
		if((!globalGood && !ofGood) || bestVarDiff > p_params.occ*sigma2*sPC)
		{
			if(idx != current->coords.size() - 1)
			{
				current->coords[idx] = current->coords[current->coords.size() - 1];
				tytr[idx] = tytr[current->coords.size() - 1];
			}
			current->coords.pop_back();
		}
	}

	int p_nSimP = current->coords.size();
	for(int idx = 0; idx < p_nSimP; ++idx)
	{
		float oldx = current->coords[idx].first;
		float oldy = current->coords[idx].second;

		int sp = p_params.sizePatch;
		std::pair<float, float> newCoord;

		if(tytr[idx])
		{
			float fnewx, fnewy;
			project(std::round(oldx), std::round(oldy), &(H[(frame-1)*nparams]), fnewx, fnewy, nparams);
			newCoord.first = fnewx;
			newCoord.second = fnewy;
		}
		else
		{
            // We use the center of the patch to get the optical flow (instead of using the one of the top left pixel)
			newCoord.first = oldx + of(std::round(oldx + (float)sp/2.), std::round(oldy + (float)sp/2.), frame-1, 0);
			newCoord.second = oldy + of(std::round(oldx + (float)sp/2.), std::round(oldy + (float)sp/2.), frame-1, 1);
		}

		// Load Y1 - X0
		//      Y1 - Y0
		//      X0
		for(int c = 0, k = 0; c < i_imNoisy.sz.channels; ++c)
		for(int dx = 0; dx < sp; ++dx)
		for(int dy = 0; dy < sp; ++dy, ++k)
		{
			i_mat.inno[k*p_nSimP + idx] = (i_imNoisy(std::round(newCoord.first) + dx, std::round(newCoord.second) + dy, frame, c) - o_imFinal(std::round(oldx) + dx, std::round(oldy) + dy, frame-1, c));
			i_mat.F[k*p_nSimP + idx] = (i_imNoisy(std::round(newCoord.first) + dx, std::round(newCoord.second) + dy, frame, c) - i_imNoisy(std::round(oldx) + dx, std::round(oldy) + dy, frame-1, c));
			current->patches[k*p_nSimP + idx] = o_imFinal(std::round(oldx) + dx, std::round(oldy) + dy, frame-1, c);
		}
		current->coords[idx] = newCoord;
	}
	
	if(current->coords.empty())
	{
		current->set_destruction = true;
		return;
	}
	current->set_destruction = false;


	if(r != 0)
	{
		bool useTrick = false;
		if(p_params.noInnovTrick)
		{
			//! Parameters
			const float threshold = 2 * sigma2 * p_params.gammaT;
			const unsigned sP2  = p_params.sizePatch * p_params.sizePatch;

			//! Compute the standard deviation of the set of patches
			const float stdDev = computeStdDeviationCentered(i_mat.F, sP2, p_nSimP, p_params.channels);

			//! If we are in a region that didn't change
			if (stdDev < threshold) {
				for(int i = 0; i < sPC*sPC; ++i)
					i_mat.covMat[i] = 0;
				useTrick = true;
			}
		}
		if(!useTrick)
		{
			covarianceMatrix(i_mat.F, i_mat.covMat, p_nSimP, sPC);

			for(int i = 0; i < sPC; ++i)
				i_mat.covMat[sPC*i + i] -= 2*sigma2;
		}

		/// Update CF
		weightedSumMatrix(current->CF, i_mat.covMat, p_params.a, (1. - p_params.a)/2.);
		/// Update P
		sumMatrix(current->P, current->CF);


		i_mat.update = current->P;

		//! Compute leading eigenvectors
		int info = matrixEigs(i_mat.update, sPC, r, i_mat.covEigVals, i_mat.covEigVecs);

		i_mat.eigVecsBackup = i_mat.covEigVecs;
		//! Compute eigenvalues-based coefficients of Bayes' filter
		for (unsigned k = 0; k < r; ++k)
		{
			//if(i_mat.covEigVals[k] < EIGTHRESH * sigma2)
			//	i_mat.covEigVals[k] = 0;
			//else
				i_mat.covEigVals[k] = (i_mat.covEigVals[k] < 1e-8f) ? 0 : 1.f / ( 1. + sigma2 / i_mat.covEigVals[k] );
		}

		float *eigVecs = i_mat.covEigVecs.data();
		for (unsigned k = 0; k < r  ; ++k)
			for (unsigned i = 0; i < sPC; ++i)
				*eigVecs++ *= i_mat.covEigVals[k];

		productMatrix(i_mat.covMat,
				i_mat.eigVecsBackup,
				i_mat.covEigVecs,
				sPC, sPC, r,
				false, true);

		productMatrix(i_mat.F,
				i_mat.inno,
				i_mat.covMat,
				p_nSimP, sPC, sPC,
				false, false);

		sumMatrix(current->patches, i_mat.F);

		productMatrix(i_mat.update,
				i_mat.covMat,
				current->P,
				sPC, sPC, sPC,
				false, false);

		weightedSumMatrix(current->P, i_mat.update, 1., -1.);
	}
}


/**
 * @brief Aggregate estimates of all similar patches contained in a group.
 *
 * @param weights: subpixelic weights to be updated;
 * @param x, y: spatial location being updated;
 * @param f: index of the frame
 * @param precomputations: contains precomputation used for the subpixelic weights;
 *
 **/
void updateWeights(Video<float>& weights, float x, float y, int f, const std::vector<float>& precomputations)
{
	for(int m=x - INDMAX - 1; m <= x + INDMAX+1; m++) // only look for relevant indices
		if(m>=0 && m<weights.sz.width) // in the image
			for(int n= y - INDMAX - 1; n<= y + INDMAX+1; n++) // only look for relevant indices
				if(n>=0 && n<weights.sz.height) // in the image
				{
					float xc = x - m;
					float yc = y - n;
					float r2 = xc*xc+yc*yc;
					if(r2<RADIUS2)
					{
						float wc = precomputations[(int) round(r2*STEP)]; //find in memory the precomputed value

#if ORDER == 2
						weights(m, n, f,  0) += wc;
						weights(m, n, f,  1) += wc*xc;
						weights(m, n, f,  2) += wc*yc;
						weights(m, n, f,  3) += wc*xc*xc;
						weights(m, n, f,  4) += wc*xc*yc;
						weights(m, n, f,  5) += wc*yc*yc;
						weights(m, n, f,  6) += wc*xc*xc*xc;
						weights(m, n, f,  7) += wc*xc*xc*yc;
						weights(m, n, f,  8) += wc*xc*yc*yc;
						weights(m, n, f,  9) += wc*yc*yc*yc;
						weights(m, n, f, 10) += wc*xc*xc*xc*xc;
						weights(m, n, f, 11) += wc*xc*xc*xc*yc;
						weights(m, n, f, 12) += wc*xc*xc*yc*yc;
						weights(m, n, f, 13) += wc*xc*yc*yc*yc;
						weights(m, n, f, 14) += wc*yc*yc*yc*yc;
#elif ORDER == 1
						weights(m, n, f,  0) += wc;
						weights(m, n, f,  1) += wc*xc;
						weights(m, n, f,  2) += wc*yc;
						weights(m, n, f,  3) += wc*xc*xc;
						weights(m, n, f,  4) += wc*xc*yc;
						weights(m, n, f,  5) += wc*yc*yc;
#else
						weights(m, n, f,  0) += wc;
#endif
					}
				}
}

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
,	int frame
,	const nlkParams &p_params
, 	const std::vector<float>& precomputations
){
	//! Parameters initializations
	const unsigned chnls = p_params.channels;
	const unsigned sPx   = p_params.sizePatch;

	//! Aggregate estimates
	for(int g = 0; g < groups.size(); ++g)
	{
		int p_nSimP = groups[g]->coords.size();
		// Aggregate the patches that were initially in the group
		for(int n = 0; n < p_nSimP; ++n)
		{
			float newx = groups[g]->coords[n].first;
			float newy = groups[g]->coords[n].second;

			// Aggregate
			for(int x=newx - INDMAX - 1; x <= newx + INDMAX+1; x++) // only look for relevant indices
			for(int y=newy - INDMAX - 1; y <= newy + INDMAX+1; y++) // only look for relevant indices
			{
				float xc = newx - x;
				float yc = newy - y;
				float r2 = xc*xc+yc*yc;
				if(r2<RADIUS2)
				{
					float wc = precomputations[(int) round(r2*STEP)]; //find in memory the precomputed value

					for (unsigned c = 0, k = 0; c < chnls; c++)
					for (unsigned px = 0; px < sPx; px++)
					for (unsigned py = 0; py < sPx; py++, k++)
						if((x+px)>=0 && (x+px)<weightsSub.sz.width && (y+py)>=0 && (y+py)<weightsSub.sz.height) // in the image
						{
#if ORDER == 2
							weightsSub(x+px, y+py, frame, 15 + 6*c) += groups[g]->patches[k * p_nSimP + n]*wc;
							weightsSub(x+px, y+py, frame, 16 + 6*c) += groups[g]->patches[k * p_nSimP + n]*wc*xc;
							weightsSub(x+px, y+py, frame, 17 + 6*c) += groups[g]->patches[k * p_nSimP + n]*wc*yc;
							weightsSub(x+px, y+py, frame, 18 + 6*c) += groups[g]->patches[k * p_nSimP + n]*wc*xc*xc;
							weightsSub(x+px, y+py, frame, 19 + 6*c) += groups[g]->patches[k * p_nSimP + n]*wc*xc*yc;
							weightsSub(x+px, y+py, frame, 20 + 6*c) += groups[g]->patches[k * p_nSimP + n]*wc*yc*yc;
#elif ORDER == 1
							weightsSub(x+px, y+py, frame, 6 + 3*c) += groups[g]->patches[k * p_nSimP + n]*wc;
							weightsSub(x+px, y+py, frame, 7 + 3*c) += groups[g]->patches[k * p_nSimP + n]*wc*xc;
							weightsSub(x+px, y+py, frame, 8 + 3*c) += groups[g]->patches[k * p_nSimP + n]*wc*yc;
#else
							weightsSub(x+px, y+py, frame, 1 +c) += groups[g]->patches[k * p_nSimP + n]*wc;

#endif
						}
				}
			}

			for (unsigned c = 0, k = 0; c < chnls; c++)
			for (unsigned px = 0; px < sPx; px++)
			for (unsigned py = 0; py < sPx; py++, k++)
				io_im(std::round(newx)+px, std::round(newy)+py, frame, c) += groups[g]->patches[k * p_nSimP + n];
		}

	}
}

/**
 * @brief Compute the final weighted aggregation (subpixelic version). 
 * This function was provided in most part by Thibaud Briand as part 
 * of his work presented in "Briand, Thibaud. "Low Memory Image 
 * Reconstruction Algorithm from RAW Images." 2018 IEEE 13th Image, 
 * Video, and Multidimensional Signal Processing Workshop (IVMSP). 
 * IEEE, 2018."
 *
 * @param i_im: will contain the final video;
 * @param i_weight: associated weight for each estimate of pixels.
 * @param frame: index of the frame on which to do aggregation
 *
 * @return : none.
 **/
void computeWeightedAggregationSub(
	Video<float> &i_im
,	const Video<float> &i_weight
, 	int frame
){
	for (unsigned y = 0; y < i_im.sz.height  ; y++)
	for (unsigned x = 0; x < i_im.sz.width   ; x++)
	{
#if ORDER == 2
		int test = 1;
		double A[36];

		A[35] = i_weight(x,y,frame, 0);
		A[34] = i_weight(x,y,frame, 1);
		A[33] = i_weight(x,y,frame, 2);
		A[32] = i_weight(x,y,frame, 3);
		A[31] = i_weight(x,y,frame, 4);
		A[30] = i_weight(x,y,frame, 5);

		A[29] = i_weight(x,y,frame, 1);
		A[28] = i_weight(x,y,frame, 3);
		A[27] = i_weight(x,y,frame, 4);
		A[26] = i_weight(x,y,frame, 6);
		A[25] = i_weight(x,y,frame, 7);
		A[24] = i_weight(x,y,frame, 8);

		A[23] = i_weight(x,y,frame, 2);
		A[22] = i_weight(x,y,frame, 4);
		A[21] = i_weight(x,y,frame, 5);
		A[20] = i_weight(x,y,frame, 7);
		A[19] = i_weight(x,y,frame, 8);
		A[18] = i_weight(x,y,frame, 9);

		A[17] = i_weight(x,y,frame, 3);
		A[16] = i_weight(x,y,frame, 6);
		A[15] = i_weight(x,y,frame, 7);
		A[14] = i_weight(x,y,frame, 10);
		A[13] = i_weight(x,y,frame, 11);
		A[12] = i_weight(x,y,frame, 12);

		A[11] = i_weight(x,y,frame, 4);
		A[10] = i_weight(x,y,frame, 7);
		A[9]  = i_weight(x,y,frame, 8);
		A[8]  = i_weight(x,y,frame, 11);
		A[7]  = i_weight(x,y,frame, 12);
		A[6]  = i_weight(x,y,frame, 13);

		A[5]  = i_weight(x,y,frame, 5);
		A[4]  = i_weight(x,y,frame, 8);
		A[3]  = i_weight(x,y,frame, 9);
		A[2]  = i_weight(x,y,frame, 12);
		A[1]  = i_weight(x,y,frame, 13);
		A[0]  = i_weight(x,y,frame, 14);


		// cholesky decomposition A=L*L' where L is lower triangular
		double L[36]={0};
		double factor = sqrt(A[0]);
		if( factor > 0 )
			L[0] = factor;
		else
			test = 0;

		for (int i=1; i<6; i++)
			L[6*i]= A[6*i]/factor;
		for (int j=1; j<6; j++) {
			double sum=0;
			for (int k=0;k<j;k++) sum+= L[j*6+k]*L[j*6+k];
			factor = sqrt(A[6*j+j] - sum);
			if( factor > 0) {
				double factor2 =  1/factor;
				L[6*j+j] = factor;
				for (int i=j+1; i<6; i++) {
					double sum2=0;
					for (int k=0;k<j;k++) sum2+=L[6*i+k]*L[6*j+k];
					L[6*i+j]= (A[6*i+j] - sum2)*factor2;
				}
			}
			else
				test = 0;
		}

		double B[6];
		double y1[6]={0};
		for (unsigned c = 0; c < i_im.sz.channels; c++)
		{
			// compute B
			for (int i=0; i<6;i++) B[5-i] = i_weight(x,y,frame, 15+i + 6*c);

			// the system is B=L*L'*x
			// we first compute y such that B=L*y
			if( L[0] > 0 )
				y1[0]=B[0]/L[0];
			else
				test = 0;

			for (int i=1;i<6;i++)
			{
				double sum = 0;
				for (int j=0;j<i;j++) sum += L[6*i+j]*y1[j];
				if( L[6*i+i]> 0 )
					y1[i]=(B[i]-sum)/L[6*i+i];
				else
					test = 0;
			}

			// then y=L'*x with L' upper triangular so the coefficient x[5] is computed with a simple division
			if( L[35] > 0)
				i_im(x,y,frame,c) = y1[5]/L[35];
			else
				test = 0;

			if(test == 0) // Failure in the previous computations. Use the zero order.
				i_im(x,y,frame,c) = i_weight(x,y,frame, 15 + 6*c) / i_weight(x,y,frame, 0);
		}


#elif ORDER == 1
		float det, m1, m2, m3;
		m1 = i_weight(x,y,frame, 5)*i_weight(x,y,frame, 3) - i_weight(x,y,frame, 4)*i_weight(x,y,frame, 4);
		m2 = -(i_weight(x,y,frame, 5)*i_weight(x,y,frame, 1) - i_weight(x,y,frame, 2)*i_weight(x,y,frame, 4));
		m3 = i_weight(x,y,frame, 4)*i_weight(x,y,frame, 1) - i_weight(x,y,frame, 2)*i_weight(x,y,frame, 3);
		det = i_weight(x,y,frame, 0)*m1 + i_weight(x,y,frame, 1)*m2 + i_weight(x,y,frame, 2)*m3;
		for (unsigned c = 0; c < i_im.sz.channels; c++)
		{
			if(det > 10e-10)
				i_im(x,y,frame,c) = (m1*i_weight(x,y,frame, 6 + 3*c) + m2*i_weight(x,y,frame, 7 + 3*c) + m3*i_weight(x,y,frame, 8 + 3*c)) / det;
			else
				// In this case the weight should always be > 0
				i_im(x,y,frame,c) = i_weight(x,y,frame, 6 + 3*c) / i_weight(x,y,frame, 0);
		}
#else
		for (unsigned c = 0; c < i_im.sz.channels; c++)
			i_im(x,y,frame,c) = i_weight(x,y,frame, 1+c) / i_weight(x,y,frame, 0);
#endif

	}
}

/**
 * @brief Compute the final weighted aggregation.
 *
 * i_im: image of reference, when the weight if null;
 * io_im: will contain the final image;
 * i_weight: associated weight for each estimate of pixels (9 channels for the first order).
 *
 * @return : none.
 **/
void computeWeightedAggregation(
	Video<float> &i_im
,	const Video<float> &i_weight
, 	int frame
){
	for (unsigned c = 0; c < i_im.sz.channels; c++)
	for (unsigned y = 0; y < i_im.sz.height  ; y++)
	for (unsigned x = 0; x < i_im.sz.width   ; x++)
		// In this case the weight should always be > 0
		i_im(x,y,frame,c) /= i_weight(x,y,frame);
}
