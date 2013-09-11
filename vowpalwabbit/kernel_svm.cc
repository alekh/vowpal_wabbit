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
#include <map>

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

    VW::flat_example** pool;
    double lambda;

    void* kernel_params;
    size_t kernel_type;
    
    vw* all;
  };

  void save_load_svm_model(svm_params* params, io_buf& model_file, bool read, bool text) {  
    svm_model* model = params->model;
    //TODO: check about initialization

    if(read) 
      

    bin_text_read_write_fixed(model_file,(char*)&model->num_support, sizeof(model->num_support), 
			      "", read, "", 0, text);
    
    for(uint32_t i = 0;i < model->num_support;i++) 
      VW::save_load_flat_example(model_file, read, model->support_vec[i]);
    
    bin_text_read_write_fixed(model_file, (char*)model->alpha.begin, model->num_support*sizeof(float),
			      "", read, "", 0, text);
    bin_text_read_write_fixed(model_file, (char*)model->delta.begin, model->num_support*sizeof(float),
			      "", read, "", 0, text);        
  }

  void save_load(void* data, io_buf& model_file, bool read, bool text) {  
    svm_params* params = (svm_params*) data;

    save_load_svm_model(params, model_file, read, text);
    
  }
  
  double linear_kernel(const VW::flat_example* fec1, const VW::flat_example* fec2) {

    double dotprod = 0;
    
    feature* ec2f = fec2->feature_map;
    uint32_t ec2pos = 0;          
    uint32_t idx1 = 0, idx2 = 0;
    
    for (feature* f = fec1->feature_map; idx1 < fec1->feature_map_len && idx2 < fec2->feature_map_len ; f++, idx1++) {
      uint32_t ec1pos = f->weight_index;
      if(ec1pos < ec2pos) continue;
      else if(ec1pos == ec2pos) {
	dotprod += f->x*ec2f->x;
	ec2f++;
	idx2++;
	if(idx2 < fec2->feature_map_len)
	  ec2pos = ec2f->weight_index;
      }
      else 
	while(ec1pos > ec2pos && idx2 < fec2->feature_map_len) {
	  ec2f++;
	  idx2++;
	  if(idx2 < fec2->feature_map_len)
	    ec2pos = ec2f->weight_index;
	}
    }    
      
    return dotprod;
  }

  double rbf_kernel(const VW::flat_example* fec1, const VW::flat_example* fec2, double bandwidth) {
    return 0;
  }

  double kernel_function(const VW::flat_example* fec1, const VW::flat_example* fec2, void* params, size_t kernel_type) {
    switch(kernel_type) {
    case SVM_KER_RBF:
      return rbf_kernel(fec1, fec2, *(double*)params);
    case SVM_KER_LIN:
      return linear_kernel(fec1, fec2);
    }
    return 0;
  }

  double dense_dot(double* v1, v_array<float> v2, size_t n) {
    double dot_prod = 0.;
    for(size_t i = 0;i < n;i++)
      dot_prod += v1[i]*v2[i];
    return dot_prod;
  }

  
  void compute_inprods(svm_params* params, const VW::flat_example* fec, double* inprods) {
    svm_model* model = params->model;
    for(size_t i = 0 ;i < model->num_support;i++)
      inprods[i] = kernel_function(fec, model->support_vec[i], params->kernel_params, params->kernel_type);
    
  }

  void predict(svm_params* params, VW::flat_example** ec_arr, double* scores, size_t n) { 
    svm_model* model = params->model;
    double* inprods = new double[model->num_support];
    
    
    for(size_t i = 0;i < n; i++) {
      compute_inprods(params, ec_arr[i], inprods);
      scores[i] = dense_dot(inprods, model->alpha, model->num_support)/params->lambda;
    }
    delete[] inprods;
  }
      
  size_t suboptimality(svm_model* model, double* subopt) {

    int max_pos = 0;

    for(size_t i = 0;i < model->num_support;i++) {
      label_data* ld = (label_data*)(model->support_vec[i]->ld);
      double tmp = model->alpha[i]*ld->label;
      
      
      double max_val = 0;
      
      if((tmp < ld->weight && model->delta[i] < 0) || (tmp > 0 && model->delta[i] > 0)) 
	subopt[i] = fabs(model->delta[i]);
      else
	subopt[i] = 0;

	if(subopt[i] > max_val) {
	  max_val = subopt[i];
	  max_pos = i;
	}
      }    
    return max_pos;
  }  

  void remove(svm_params* params, int pos) {
    //cerr<<"remove\n";
    svm_model* model = params->model;
    int num_support = model->num_support;
    
    if(pos < num_support - 1) {
      model->support_vec[pos] = model->support_vec[num_support - 1];      
      model->alpha[pos] = model->alpha[num_support - 1];
      model->delta[pos] = model->delta[num_support - 1];      
    }
    model->support_vec.pop();
    VW::free_flat_example(model->support_vec[num_support-1]);    
    model->alpha.pop();
    model->delta.pop();
    model->num_support--;
  }

  int add(svm_params* params, VW::flat_example* fec) {
    svm_model* model = params->model;
    model->num_support++;
    model->support_vec.push_back(fec);
    model->alpha.push_back(0.);
    model->delta.push_back(0.);
    return (model->support_vec.size()-1);
  }

  void update(svm_params* params, int pos) {

    //cerr<<"Updating model "<<pos<<" "<<params->model->num_support<<endl;
        
    svm_model* model = params->model;
    double* inprods = new double[model->num_support];
    VW::flat_example* fec = model->support_vec[pos];
    label_data* ld = (label_data*) fec->ld;
    compute_inprods(params, fec, inprods);
    
    //cerr<<"Computed inprods\n";

    model->delta[pos] = dense_dot(inprods, model->alpha, model->num_support)*ld->label/params->lambda - 1;
    double alpha_old = model->alpha[pos];
    model->alpha[pos] = 0.;

    double proj = dense_dot(inprods, model->alpha, model->num_support)*ld->label;
    double ai = (params->lambda - proj)/inprods[pos];
    
    //cerr<<model->num_support<<" "<<proj<<" "<<ai<<" "<<pos<<" "<<ld->label<<" ";

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
    
    for(size_t i = 0;i < model->num_support; i++) {
      label_data* ldi = (label_data*) model->support_vec[i]->ld;
      model->delta[i] += diff*inprods[i]*ldi->label/params->lambda;
    }
    
    //cerr<<ai<<" "<<diff<<endl;
    
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
    
    bool* train_pool = new bool[params->pool_size];
    
    double* scores = new double[params->pool_size];
    predict(params, params->pool, scores, params->pool_size);

    for(size_t i = 0;i < params->pool_size;i++) {
      float label = ((label_data*)params->pool[i]->ld)->label;
      double loss = max(0., 1 - scores[i]*label);
      //cerr<<"Loss = "<<loss<<" "<<scores[i]<<" "<<label<<" "<<params->model->num_support<<endl;
      params->all->sd->sum_loss += loss;
    }
      
    if(params->active) {           
      if(params->active_pool_greedy) { 
	map<double, int, std::greater<double> > scoremap;
	for(size_t i = 0;i < params->pool_size; i++)
	  scoremap[scores[i]] = i;

	map<double, int, std::greater<double> >::iterator iter = scoremap.begin();
	
	for(size_t train_size = 1;iter != scoremap.end() && train_size <= params->subsample;train_size++) {
	  train_pool[iter->second] = 1;
	  iter++;	  
	}
      }
      else {

	for(size_t i = 0;i < params->pool_size;i++) {
	  double queryp = 2.0/(1.0 + exp(params->all->active_c0*fabs(scores[i])*pow(params->pool[i]->example_counter,0.5)));
	  if(rand() < queryp) {
	    VW::flat_example* fec = params->pool[i];
	    label_data* ld = (label_data*)fec->ld;
	    ld->weight *= 1/queryp;
	    train_pool[i] = 1;
	  }
	}
      }
      delete[] scores;
    }

    if(params->all->training) {
      
      svm_model* model = params->model;
      
      for(size_t i = 0;i < params->pool_size;i++) {
	int model_pos;
	if(params->active)
	  if(train_pool[i])
	    model_pos = add(params, params->pool[train_pool[i]]);
	  else
	    VW::free_flat_example(params->pool[i]);
	else
	  model_pos = add(params, params->pool[i]);
	//cerr<<"Added: "<<&(model->support_vec[model_pos])<<endl;
	update(params, model_pos);
	
	for(size_t j = 0;j < params->reprocess;j++) {
	  if(model->num_support == 0) break;
	  //cerr<<"reprocess\n";
	  double* subopt = new double[model->num_support];
	  size_t max_pos = suboptimality(model, subopt);
	  
	    if(subopt[max_pos] > 0)
	      update(params, max_pos);
	    
	    delete[] subopt;
	}
      }

    }

    delete[] scores;
    delete[] train_pool;
  }

  void learn(void* d, example* ec) {
    svm_params* params = (svm_params*)d;
    
    VW::flat_example* fec = VW::flatten_example(*(params->all),ec);
    //cerr<<fec<<endl;
    
    if(fec)
      assert(fec->in_use);
    
    if(fec) {
      
      params->pool[params->pool_pos] = fec;
      params->pool_pos++;
      
      if(params->pool_pos == params->pool_size) {
	train(params);
	params->pool_pos = 0;
      }
	
      
    }
  }

  void driver(vw* all, void* data) {
    example* ec = NULL;
   
    cerr<<"Starting kernel SVM\n";
 
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

  void free_svm_model(svm_model* model)
  {
    //cerr<<"Freeing the model "<<model->num_support<<endl;
    for(size_t i = 0;i < model->num_support; i++) {
      //cerr<<"Example counter = "<<model->support_vec[i]->example_counter<<" "<<model->support_vec[i]<<" "<<sizeof(VW::flat_example)<<endl; 
       VW::free_flat_example(model->support_vec[i]);
     }
    //cerr<<"Done freeing support vectors\n";

    model->support_vec.delete_v();
    model->alpha.delete_v();
    model->delta.delete_v();
    //cerr<<"Freed all the v_arrays in model\n";
    free(model);
    //cerr<<"Freed the model\n";
  }

  void finish(void* d) {
    //cerr<<"Entering finish\n";
    svm_params* params = (svm_params*) d;
    free(params->pool);

    //cerr<<"Done freeing pool\n";

    free_svm_model(params->model);
    //cerr<<"Done freeing model\n";
    if(params->kernel_params) free(params->kernel_params);
    //cerr<<"Done freeing kernel params\n";
    free(params);
    //cerr<<"Done with finish \n";
  }


  learner setup(vw &all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file) {
    svm_params* params = (svm_params*) calloc(1,sizeof(svm_params));
    //cerr<<"In setup\n";

    params->model = (svm_model*) calloc(1,sizeof(svm_model));
    params->model->num_support = 0;
    params->model->maxdelta = 0.;
    
    po::options_description desc("KSVM options");
    desc.add_options()
      ("reprocess", po::value<size_t>(), "number of reprocess steps for LASVM");
    desc.add_options()
      ("pool_greedy", "use greedy selection on mini pools");
    desc.add_options()
      ("para_active", "do parallel active learning");
    desc.add_options()
      ("pool_size", po::value<size_t>(), "size of pools for active learning");
    desc.add_options()
      ("subsample", po::value<size_t>(), "number of items to subsample from the pool");
    desc.add_options()
      ("kernel", po::value<string>(), "type of kernel (rbf or linear (default))");
    desc.add_options()
      ("bandwidth", po::value<double>(), "bandwidth of rbf kernel");

    po::parsed_options parsed = po::command_line_parser(opts).
      style(po::command_line_style::default_style ^ po::command_line_style::allow_guessing).
      options(desc).allow_unregistered().run();
    opts = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    po::notify(vm);

    po::parsed_options parsed_file = po::command_line_parser(all.options_from_file_argc,all.options_from_file_argv).
      style(po::command_line_style::default_style ^ po::command_line_style::allow_guessing).
      options(desc).allow_unregistered().run();
    po::store(parsed_file, vm_file);
    po::notify(vm_file);
    
    //cerr<<"Done parsing args\n";

    params->all = &all;
    
    if(vm_file.count("reprocess"))
      params->reprocess = vm_file["reprocess"].as<std::size_t>();
    else if(vm.count("reprocess"))
      params->reprocess = vm["reprocess"].as<std::size_t>();
    else 
      params->reprocess = 1;

    params->active = all.active_simulation;        
    if(params->active) {
      if(vm.count("pool_greedy"))
	params->active_pool_greedy = 1;
      if(vm.count("para_active"))
	params->para_active = 1;
    }
    
    if(vm_file.count("pool_size"))
      params->pool_size = vm_file["pool_size"].as<std::size_t>();
    else if(vm.count("pool_size")) 
      params->pool_size = vm["pool_size"].as<std::size_t>();
    else
      params->pool_size = 1;
    
    params->pool = (VW::flat_example**)calloc(params->pool_size, sizeof(VW::flat_example*));
    params->pool_pos = 0;
    
    if(vm_file.count("subsample"))
      params->subsample = vm["subsample"].as<std::size_t>();
      else if(vm.count("subsample"))
	params->subsample = vm["subsample"].as<std::size_t>();
      else
	params->subsample = 1;
    
    params->lambda = all.l2_lambda;

    std::string kernel_type;

    if(vm_file.count("kernel") || vm.count("kernel")) {
	
      if(vm_file.count("kernel")) {
	kernel_type = vm_file["kernel"].as<std::string>();
	if(vm.count("kernel") && kernel_type.compare(vm["kernel"].as<string>()) != 0) 
	  cerr<<"You selected a different kernel with --kernel than the one in regressor file. Pursuing with loaded value of "<<kernel_type<<endl;
      }
      else 
	kernel_type = vm["kernel"].as<std::string>();
    }
    else
      kernel_type = string("linear");
    
    if(!vm_file.count("kernel")) {
      std::stringstream ss;
      ss <<" --kernel "<< kernel_type;
      all.options_from_file.append(ss.str());
    }

    if(kernel_type.compare("rbf") == 0) {
      params->kernel_type = SVM_KER_RBF;
      double bandwidth = 1.;
      if(vm.count("bandwidth") || vm_file.count("bandwidth")) {
	if(vm_file.count("bandwidth")) {
	  bandwidth = vm_file["bandwidth"].as<double>();
	  if(vm.count("bandwidth") && bandwidth != vm["bandwidth"].as<double>())
	    cerr<<"You specified a different bandwidth with --bandwidth than the one in the regressor file. Pursuing with the loaded value of "<<bandwidth<<endl;
	} 
	else {
	  std::stringstream ss;
	  bandwidth = vm["bandwidth"].as<double>();	
	  ss<<" --bandwidth "<<bandwidth;
	  all.options_from_file.append(ss.str());
	}
      }
      params->kernel_params = &bandwidth;
    }      
    else
      params->kernel_type = SVM_KER_LIN;            
    
    all.l.finish();
    sl_t sl = {params, save_load};
    learner ret(params, driver, learn, finish, sl);
    //cerr<<"Done with setup\n";
    return ret;
  }    


}
