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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <string>
#include <sstream>

#include "cmd_option.h"
#include "nlkalman/Utilities.h"
#include "nlkalman/nlkalman.h"

using namespace std;

/**
 * @file   main.cpp
 * @brief  Main executable file
 *
 * @author THIBAUD EHRET  <ehret.thibaud@gmail.com>
 **/

int main(int argc, char **argv)
{
	clo_usage("NL-Kalman-Bayes video denoising");
	clo_help(" NOTE: Input (<) and output (>) sequences are specified by their paths in printf format.\n");

	//! Paths to input/output sequences
	using std::string;
	const string  input_path = clo_option("-i"    , ""              , "< Input sequence");
	const string  of_path    = clo_option("-of"   , ""              , "< Input optical flow");
	const string  noisy_path = clo_option("-nisy" , "noisy_%03d.png", "> Noisy sequence");
	const string  final_path = clo_option("-deno" , "deno_%03d.png" , "> Denoised sequence");
	const string  diff_path  = clo_option("-diff" , "diff_%03d.png" , "> Difference sequence");

	const unsigned firstFrame = clo_option("-f", 0, "< Index of the first frame");
	const unsigned lastFrame  = clo_option("-l", 0, "< Index of the last frame");
	const unsigned frameStep  = clo_option("-s", 1, "< Frame step");

	//! General parameters
	const float sigma = clo_option("-sigma", 0., "< Noise of standard deviation sigma");
	const float a = clo_option("-a", 0.9, "< Memory parameter");
	const float occ = clo_option("-occ", 8.25, "< Occlusion threshold");
	const bool add_noise = (bool) clo_option("-add_noise", true, "< Add the noise to the input");
	const bool verbose  = (bool) clo_option("-verbose"     , true , "> verbose output");

	//! NLB parameters
	const int space_search  = clo_option("-wx",15, "> Search window spatial radius");
	const int patch_size    = clo_option("-px", 8, "> Spatial patch size");
	const int num_patches   = clo_option("-np",64, "> Number of similar patches");
	const int rank          = clo_option("-r" ,16, "> Rank of covariance matrix");


    //! The following parameters are not described in the ICIP article but are described in the joint IPOL article 
	const string  stm_path   = clo_option("-stm"  , ""              , "< stabilization matrices");
	const string  sub_path   = clo_option("-sub"  , "sub_%03d.png"  , "> subpixelic sequence");
	const bool flatAreaTrick  = (bool) clo_option("-flat"     , false , "< use the flat area trick");
	const bool noInnovTrick  = (bool) clo_option("-innov"     , false , "> use the no innovation trick");

	//! Check inputs
	if (input_path == "")
	{
		fprintf(stderr, "%s: no input sequence.\nTry `%s --help' for more information.\n",
				argv[0], argv[0]);
		return EXIT_FAILURE;
	}

	//! Declarations
	Video<float> original, noisy, final, subfinal, diff, of;

	//! Load input videos
	original.loadVideo(input_path, firstFrame, lastFrame, frameStep);

	//! Add noise
	if (add_noise)
	{
		VideoUtils::addNoise(original, noisy, sigma, verbose);

		//! Save noisy video
		if (verbose) printf("Saving noisy video\n");
		noisy.saveVideo(noisy_path, firstFrame, frameStep);

        if(patch_size == 0)
            return EXIT_SUCCESS;
	}
	else
		noisy = original;

	of.loadFullFlow(of_path, firstFrame, lastFrame-1, frameStep);
	//! load stabilization matrices
	float* H = NULL;
	int nparams, ntransforms, nx, ny;
	read_transforms(stm_path.c_str(), H, nparams, ntransforms, nx, ny);

	//! Check if every size are consistent
	if((H != NULL) && ((ntransforms !=  original.sz.frames) || (nx != original.sz.width) || (ny != original.sz.height)))
	{
		printf("Transformation matrices are not consistent with the provided video, video is %d %d %d, transformation is %d %d %d\nExiting !\n", original.sz.width, original.sz.height, original.sz.frames, nx, ny, ntransforms);
		return EXIT_FAILURE;
	}


	//! Denoising
	if (verbose) printf("Running NL-Kalman on the noisy video\n");

	//! Compute denoising default parameters
	nlkParams prms;
	initializeNlkParameters(prms, sigma, noisy.sz, verbose, flatAreaTrick, noInnovTrick, a, occ, rank);
	setSizeSearchWindow(prms, (unsigned)space_search);
	setSizePatch(prms, noisy.sz, (unsigned)patch_size);
	setNSimilarPatches(prms, (unsigned)num_patches);

    //! Print parameters
	if (verbose) printNlkParameters(prms);

	//! Run denoising algorithm
	nlKalman(noisy, final, subfinal, of, H, nparams, prms); 
	printf("Computing done, saving results\n");

	//! Compute PSNR and RMSE
	float final_psnr = -1, final_rmse = -1;
	VideoUtils::computePSNR(original, final, final_psnr, final_rmse);

	if (verbose)
	{
	    printf("final PSNR =\t%f\tRMSE =\t%f\n", final_psnr, final_rmse);
	}

	//! Write measures
	writingMeasures("measures.txt", sigma, final_psnr, final_rmse);

	//! Compute Difference
	VideoUtils::computeDiff(original, final, diff, sigma);

	//! Save output sequences
	if (verbose) printf("Saving output sequences\n");
	final.saveVideo(final_path, firstFrame, frameStep);
	diff .saveVideo( diff_path, firstFrame, frameStep);
	subfinal.saveVideo( sub_path, firstFrame, frameStep);

	if(H != NULL)
		delete[] H;

	if (verbose) printf("Done\n");
	return EXIT_SUCCESS;
}
