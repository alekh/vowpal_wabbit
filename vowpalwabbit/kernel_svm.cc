/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include <fstream>
#include <sstream>
#include <float.h>
#ifdef _WIN32
#include <WinSock2.h>
#else
#include <netdb.h>
#endif
#include <string.h>
#include <stdio.h>
#include <assert.h>

#if defined(__SSE2__) && !defined(VW_LDA_NO_SSE)
#include <xmmintrin.h>
#endif

#include "parse_example.h"
#include "constant.h"
#include "sparse_dense.h"
#include "gd.h"
#include "kernel_svm.h"
#include "cache.h"
#include "simple_label.h"
#include "accumulate.h"
#include "learner.h"
#include "vw.h"

using namespace std;

namespace KSVM
{

  struct svm_params{
    size_t current_pass;
    bool active;

    svm_model* model;
    
    vw* all;
  }

  void save_load(void* data, io_buf& model_file, bool read, bool text) {
  }


  void learn(void* d, example* ec) {
    svm_params* params = (gd*)d;
    vw* all = params->all;
    
    assert(ec->in_use);
    if (ec->end_pass) {
      //save model here
    }
  }

  void driver(vw* all, void* data) {
    example* ec = NULL;
    
    while ( true ) {
	if ((ec = VW::get_example(all->p)) != NULL)//semiblocking operation.
	  {
	    learn(data, ec);
	    return_simple_example(*all, ec);
	  }
      else if (parser_done(all->p))
	return;
      else 
	;//busywait when we have predicted on all examples but not yet trained on all.
      }
  }


  learner setup(vw &all) {
    svm_params* params = (svm_params*) calloc(1,sizeof(svm_params));

    params->model = (svm_model*) calloc(1,sizeof(svm_model));
    params->model->num_support = 0;
    params->model->maxdelta = 0.;
    
    svm_params->all = all;
    svm_params->active = all.active;    
    
    sl_t sl = {ksvm, save_load};
    learner ret(ksvm, driver, learn, finish, sl);
    
    return ret;
  }  

}
