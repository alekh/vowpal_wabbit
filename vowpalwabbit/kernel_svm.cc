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
#include <map.h>

using namespace std;

namespace KSVM
{
  
  struct svm_params{
    size_t current_pass;
    bool active;
    bool active_pool_greedy;
    bool para_active;

    size_t pool_size;
    size_t pool_pos;
    size_t subsample; //NOTE: Eliminating subsample to only support 1/pool_size
    size_t reprocess;

    svm_model* model;

    example** pool;
    double lambda;

    void* kernel_params;
    double (*kernel_function) (example*, example*, void*);
    
    vw* all;
  };

  void save_load(void* data, io_buf& model_file, bool read, bool text) {   //TODO
  }
  
  double dense_dot(double* v1, double* v2, int n) {
    double dot_prod = 0.;
    for(int i = 0;i < n;i++)
      dot_prod += v1[i]*v2[i];
    return dot_prod;
  }

  
  void compute_inprods(svm_params* params, example* ec, double* inprods) {
    //TODO: Update with a flatten operation
    
    svm_model* model = params->model;
    for(int = 0 ;i < model->num_support;i++)
      inprods[i] = params->kernel_function(ec, model->support_vec[i], params->kernel_params);
    
  }

  void predict(svm_params* params, example** ec_arr, double* scores, int n) { 
    svm_model* model = params->model;
    double* inprods = new double[model->num_support];
    
    
    for(int i = 0;i < n; i++) {
      compute_inprods(params, ec_arr[i], inprods);
      scores[i] = dense_dot(inprods, model->alpha, model->num_support)/params->lambda;
    }
    delete[] inprods;
  }
      
  size_t suboptimality(svm_model* model, double* subopt) {
    for(int i = 0;i < model->num_support;i++) {
      label_data* ld = (label_data*)(model->support_vec[i]->ld);
      double tmp = model->alpha[i]*ld->label;
      
      int max_pos = 0;
      double max_val = 0;
      
      if((tmp < ld->weight && model->delta[i] < 0) || (tmp > 0 && model.delta[i] > 0)) {
	subopt[i] = fabs(model.delta[i]);
      else
	subopt[i] = 0;

	if(subopt[i] > max_val) {
	  max_val = subopt[i];
	  max_pos = i;
	}
      }
    }
    return max_pos;
  }  

  void remove(svm_params* params, int pos) {
    svm_model* model = params->model;
    int num_support = model->num_support;
    
    if(pos < num_support - 1) {
      model->support_vec[pos] = model->support_vec[num_support - 1];
      model->alpha[pos] = model->alpha[num_support - 1];
      model->delta[pos] = model->delta[num_support - 1];
    }
    model->support_vec.pop();
    model->alpha.pop();
    model->delta.pop();
    model->num_support--;
  }

  void add(svm_params* params, example* ec) {
    svm_model* model = params->model;
    model->num_support++;
    model->support_vec.push_back(ec);
    model->alpha.push_back(0.);
    model->delta.push_back(0.);
  }

  void update(svm_params* params, example* ec, int pos) {
    
    label_data* ld = (label_data*) ec->ld;
    svm_model* model = params->model;
    double* inprods = new double[model->num_support];
    compute_inprods(params, ec, inprods);
    
    model->delta[pos] = dense_dot(inprods, model->alpha, model->num_support)*ld->label/params->lambda - 1;
    double alpha_old = model->alpha[pos];
    model->alpha[pos] = 0.;

    double proj = dense_dot(inprods, model->alpha, model->num_support)*ld->label;
    double ai = (params->lambda - proj)/inprods[pos];

    if(ai > ld->weight)				
      ai = ld->weight;
    else if(ai < 0)
      ai = 0;

    ai *= ld->label;
    double diff = ai - alpha_old;
    if(fabs(diff) > 1.) {
      diff = sign(diff);
      ai = alpha_old + diff;
    }
    
    for(int i = 0;i < model->num_support; i++) {
      label_data* ldi = (label_data*) model->ec[i]->ld;
      model.delta[i] += diff*inprods[i]*ldi.label/params.lambda;
    }

    if(ai == 0)
      remove(params, pos);
    else
      model->alpha[pos] = ai;
    
    double* subopt = new double[model->num_support];
    size_t max_pos = suboptimality(model, subopt);
    model->maxdelta = subopt[max_pos];
    delete[] subopt;
    delete[] inprods;
  }

  void train(svm_params* params) {
    
    size_t train_size = 0;
    example** train_pool;
    
    if(params->active_simulation) {      
      double* scores = new double[params->pool_size];
      predict(params->model, params->pool, scores, params->pool_size);

      if(params->active_pool_greedy) {
	map<double, int, std::greater<double> > scoremap;
	for(int i = 0;i < params->pool_size; i++)
	  scoremap[scores[i]] = i;

	map<double, int, std::greater<double> >::iterator iter = scoremap.begin();
	
	for(train_size = 1;iter != scoremap.end() && train_size <= params.subsample;train_size++) {
	  train_pool[train_size-1] = params->pool[iter->second];
	  iter++;	  
	}
      }
      else { //TODO: figure out what all to implement here, should IWAL be supported?

	for(int i = 0;i < params->pool_size;i++) {
	  double queryp = 2.0/(1.0 + exp(params->all->active_c0*fabs(score[i])*pow(ec[i]->t,0.5)));
	  if(rand() < queryp) {
	    example ec = params->pool[i];
	    label_data* ld = (label_data*)ec->ld;
	    ld->weight *= 1/queryp;
	    train_pool[train_size] = ec;
	    train_size++;
	  }
	}
      }
      delete[] scores;
    }
    else {
      train_size = pool_size;
      train_pool = params->pool;
    }

    if(params->all->train) {
      
      svm_model* model = params->model;
      
      for(int i = 0;i < train_size;i++) {
	int model_pos = add(model, train_pool[i]);
	update(params, train_pool[i], model_pos);
	
	for(int j = 0;j < params->reprocess;j++) {
	  if(model->num_support == 0) break;
	  
	  double* subopt = new double[model->num_support];
	  size_t max_pos = suboptimality(model, subopt);

	  if(subopt[max_pos] > 0)
	    update(params, train_pool[i], max_pos);
	  
	  delete[] subopt;
	}
      }

    }
    
  }

  void learn(void* d, example* ec) {
    svm_params* params = (svm_params*)d;
    vw* all = params->all;

    
    
    assert(ec->in_use);
    if (ec->end_pass) {
      //save model here
    }
    
    if(!command_example(all,ec)) {
      
      params->pool[params->pool_pos] = ec;
      params->pool_pos++;
      
      if(params->pool_pos == params->pool_size) {
	train(params);
	params->pool_pos = 0;
      }
	
      
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
    
    params->all = all;
    params->reprocess = all->reprocess;
    params->active = all.active;    
    params->active_pool_greedy = all->active_pool_greedy;
    params->para_active = all->para_active;
    
    params->pool_size = all->pool_size;
    params->pool = new example*[params->pool_size];
    params->pool_pos = 0;
    params->subsample = all->subsample;

    params->lambda = all->lambda;
    
    sl_t sl = {ksvm, save_load};
    learner ret(ksvm, driver, learn, finish, sl);
    
    return ret;
  }  

  void finish(void* d) {
    svm_params* params = (svm_params*) d;
    delete[] params->pool;
    free(params);
  }

}
