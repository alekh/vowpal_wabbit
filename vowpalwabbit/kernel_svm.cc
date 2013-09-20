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
    size_t maxcache;
    //size_t curcache;

    svm_example** pool;
    double lambda;

    void* kernel_params;
    size_t kernel_type;
    
    vw* all;
  };

  static int num_kernel_evals = 0;
  static int num_cache_evals = 0;


  svm_example::svm_example(VW::flat_example *fec)
  {
    *(VW::flat_example*)this = *fec;
    memset(fec, 0, sizeof(flat_example));
  }

  svm_example::~svm_example()
  {
    krow.delete_v();
    // free flatten example contents
    VW::flat_example *fec = (VW::flat_example*)malloc(sizeof(VW::flat_example));
    *fec = *(VW::flat_example*)this;
    VW::free_flatten_example(fec); // free contents of flat example and frees fec.
  }


  float
  kernel_function(const VW::flat_example* fec1, const VW::flat_example* fec2, 
		  void* params, size_t kernel_type);

  int
  svm_example::compute_kernels(svm_params *params)
  {
    int alloc = 0;
    svm_model *model = params->model;
    int n = model->num_support;
   
    if (krow.size() < n)
      {
	//computing new kernel values and caching them
	//if(params->curcache + n > params->maxcache)
	//trim_cache(params);
	  num_kernel_evals += krow.size();
	for (int i=krow.size(); i<n; i++)
	  {	    
	    svm_example *sec = model->support_vec[i];
	    float kv = kernel_function(this, sec, params->kernel_params, params->kernel_type);
	    krow.push_back(kv);
	    alloc += 1;
	  }
      }
    else
      num_cache_evals += n;
    return alloc;
  }

  int 
  svm_example::clear_kernels()
  {
    int rowsize = krow.size();
    krow.end = krow.begin;
    krow.resize(0);
    return -rowsize;
  }

  
  static int 
  make_hot_sv(svm_params *params, int svi)
  {
    svm_model *model = params->model;    
    int n = model->num_support;
    if (svi >= model->num_support) 
      cerr << "Internal error at " << __FILE__ << ":" << __LINE__ << endl;
    // rotate params fields
    svm_example *svi_e = model->support_vec[svi];
    int alloc = svi_e->compute_kernels(params);
    float svi_alpha = model->alpha[svi];
    float svi_delta = model->delta[svi];
    for (int i=svi; i>0; --i)
      {
	model->support_vec[i] = model->support_vec[i-1]; 
	model->alpha[i] = model->alpha[i-1];
	model->delta[i] = model->delta[i-1];
      }
    model->support_vec[0] = svi_e;
    model->alpha[0] = svi_alpha;
    model->delta[0] = svi_delta;
    // rotate cache    
    for (int j=0; j<n; j++)
      {
	svm_example *e = model->support_vec[j];
	int rowsize = e->krow.size();
	if (svi < rowsize)
	  {
	    float kv = e->krow[svi];
	    for (int i=svi; i>0; --i)
	      e->krow[i] = e->krow[i-1];
	    e->krow[0] = kv;
	  }
	else 
	  {
	    float kv = svi_e->krow[j];
	    e->krow.push_back(0);
	    alloc += 1;
	    for (int i=e->krow.size()-1; i>0; --i)
	      e->krow[i] = e->krow[i-1];
	    e->krow[0] = kv;
	  }
      }
    return alloc;
  }

  static int 
  trim_cache(svm_params *params)
  {
    size_t sz = params->maxcache;
    svm_model *model = params->model;
    int n = model->num_support;
    int alloc = 0;
    for (int i=0; i<n; i++)
      {
	svm_example *e = model->support_vec[i];
	sz -= e->krow.size();
	if (sz < 0)
	  alloc += e->clear_kernels();
      }
    return alloc;
  }

  void save_load_svm_model(svm_params* params, io_buf& model_file, bool read, bool text) {  
    svm_model* model = params->model;
    //TODO: check about initialization

    //cerr<<"Save load svm "<<read<<" "<<text<<endl;
    if (model_file.files.size() == 0) return;

    bin_text_read_write_fixed(model_file,(char*)&(model->num_support), sizeof(model->num_support), 
			      "", read, "", 0, text);
    //cerr<<"Read num support "<<model->num_support<<endl;
        
    VW::flat_example* fec;
    if(read)
      model->support_vec.resize(model->num_support);

    for(uint32_t i = 0;i < model->num_support;i++) {
      //cerr<<"Calling save_load_flat_example\n";      
      if(read) {
	VW::save_load_flat_example(model_file, read, fec);
	model->support_vec.push_back(new svm_example(fec));
	VW::free_flatten_example(fec);
	//cerr<<model->support_vec[i]->example_counter<<" "<<fec->example_counter<<" "<<fec<<endl;
      }
      else {
	fec = model->support_vec[i];
	VW::save_load_flat_example(model_file, read, fec);
      }
      //cerr<<model->support_vec[i]->example_counter<<":"<<model->support_vec[i]->feature_map[10].x<<endl;
      //model->support_vec.push_back(fec);
      //cerr<<ret<<" ";
    }
    //cerr<<endl;
    
    //cerr<<"Read model"<<endl;
    
    if(read)
      model->alpha.resize(model->num_support);
    bin_text_read_write_fixed(model_file, (char*)model->alpha.begin, model->num_support*sizeof(float),
			      "", read, "", 0, text);
    if(read)
      model->delta.resize(model->num_support);
    bin_text_read_write_fixed(model_file, (char*)model->delta.begin, model->num_support*sizeof(float),
			      "", read, "", 0, text);        

    // cerr<<"In save_load\n";
    // for(int i = 0;i < model->num_support;i++)
    //   cerr<<model->alpha[i]<<" ";
    // cerr<<endl;
  }

  void save_load(void* data, io_buf& model_file, bool read, bool text) {  
    svm_params* params = (svm_params*) data;
    if(text) {
      cerr<<"Not supporting readable model for kernel svm currently\n";
      return;
    }

    save_load_svm_model(params, model_file, read, text);
    
  }
  
  float linear_kernel(const VW::flat_example* fec1, const VW::flat_example* fec2) {

    float dotprod = 0;
    
    feature* ec2f = fec2->feature_map;
    uint32_t ec2pos = ec2f->weight_index;          
    uint32_t idx1 = 0, idx2 = 0;
    
    //cerr<<"Intersection ";
    int numint = 0;
    for (feature* f = fec1->feature_map; idx1 < fec1->feature_map_len && idx2 < fec2->feature_map_len ; f++, idx1++) {      
      uint32_t ec1pos = f->weight_index;      
      //cerr<<ec1pos<<" "<<ec2pos<<" "<<idx1<<" "<<idx2<<" "<<f->x<<" "<<ec2f->x<<endl;
      if(ec1pos < ec2pos) continue;

      while(ec1pos > ec2pos && idx2 < fec2->feature_map_len) {
	ec2f++;
	idx2++;
	if(idx2 < fec2->feature_map_len)
	  ec2pos = ec2f->weight_index;
      }      

      if(ec1pos == ec2pos) {	
	//cerr<<ec1pos<<" "<<ec2pos<<" "<<idx1<<" "<<idx2<<" "<<f->x<<" "<<ec2f->x<<endl;
	numint++;
	dotprod += f->x*ec2f->x;
	//cerr<<f->x<<" "<<ec2f->x<<" "<<dotprod<<" ";
	ec2f++;
	idx2++;
	//cerr<<idx2<<" ";
	if(idx2 < fec2->feature_map_len)
	  ec2pos = ec2f->weight_index;
      }
    }
    //cerr<<endl;
    //cerr<<"numint = "<<numint<<endl;
      
    return dotprod;
  }

  float rbf_kernel(const VW::flat_example* fec1, const VW::flat_example* fec2, double bandwidth) {
    float dotprod = linear_kernel(fec1, fec2);    
    //cerr<<"Bandwidth = "<<bandwidth<<endl;
    return exp(-(fec1->total_sum_feat_sq + fec2->total_sum_feat_sq - 2*dotprod)*bandwidth);
  }

  float kernel_function(const VW::flat_example* fec1, const VW::flat_example* fec2, void* params, size_t kernel_type) {
    switch(kernel_type) {
    case SVM_KER_RBF:
      return rbf_kernel(fec1, fec2, *((double*)params));
    case SVM_KER_LIN:
      return linear_kernel(fec1, fec2);
    }
    return 0;
  }

  float dense_dot(float* v1, v_array<float> v2, size_t n) {
    float dot_prod = 0.;
    for(size_t i = 0;i < n;i++)
      dot_prod += v1[i]*v2[i];
    return dot_prod;
  }

  
  void predict(svm_params* params, svm_example** ec_arr, double* scores, size_t n) { 
    svm_model* model = params->model;
    for(size_t i = 0;i < n; i++) {
      ec_arr[i]->compute_kernels(params);
      scores[i] = dense_dot(ec_arr[i]->krow.begin, model->alpha, model->num_support)/params->lambda;
    }
  }
      
  size_t suboptimality(svm_model* model, double* subopt) {

    int max_pos = 0;
    //cerr<<"Subopt ";
    double max_val = 0;
    for(size_t i = 0;i < model->num_support;i++) {
      label_data* ld = (label_data*)(model->support_vec[i]->ld);
      //cerr<<ld->weight<<endl;
      double tmp = model->alpha[i]*ld->label;                  
      
      if((tmp < ld->weight && model->delta[i] < 0) || (tmp > 0 && model->delta[i] > 0)) 
	subopt[i] = fabs(model->delta[i]);
      else
	subopt[i] = 0;

	if(subopt[i] > max_val) {
	  max_val = subopt[i];
	  max_pos = i;
	}
	//cerr<<subopt[i]<<" ";
      }    
    //cerr<<endl;
    return max_pos;
  }  

  int remove(svm_params* params, int svi) {
    svm_model* model = params->model;
    if (svi >= model->num_support) 
      cerr << "Internal error at " << __FILE__ << ":" << __LINE__ << endl;
    // shift params fields
    svm_example* svi_e = model->support_vec[svi];
    for (int i=svi; i<model->num_support-1; ++i)
      {
	model->support_vec[i] = model->support_vec[i+1];
	model->alpha[i] = model->alpha[i+1];
	model->delta[i] = model->delta[i+1];
      }
    delete svi_e;
    model->support_vec.pop();
    model->alpha.pop();
    model->delta.pop();
    model->num_support--;
    // shift cache
    int alloc = 0;
    for (int j=0; j<model->num_support; j++)
      {
	svm_example *e = model->support_vec[j];
	int rowsize = e->krow.size();
	if (svi < rowsize)
	  {
	    for (int i=svi; i<rowsize-1; i++)
	      e->krow[i] = e->krow[i+1];
	    e->krow.pop();
	    alloc -= 1;
	  }
      }
    return alloc;
  }

  int add(svm_params* params, svm_example* fec) {
    svm_model* model = params->model;
    model->num_support++;
    model->support_vec.push_back(fec);
    model->alpha.push_back(0.);
    model->delta.push_back(0.);
    return (model->support_vec.size()-1);
  }

  bool update(svm_params* params, int pos) {

    //cerr<<"Update\n";
    svm_model* model = params->model;
    bool overshoot = false;
    //cerr<<"Updating model "<<pos<<" "<<model->num_support<<" ";
    //cerr<<model->support_vec[pos]->example_counter<<endl;
    svm_example* fec = model->support_vec[pos];
    label_data* ld = (label_data*) fec->ld;
    fec->compute_kernels(params);
    float *inprods = fec->krow.begin;
    double alphaKi = dense_dot(inprods, model->alpha, model->num_support);
    model->delta[pos] = alphaKi*ld->label/params->lambda - 1;
    double alpha_old = model->alpha[pos];
    alphaKi -= model->alpha[pos]*inprods[pos];
    model->alpha[pos] = 0.;
    
    double proj = alphaKi*ld->label;
    double ai = (params->lambda - proj)/inprods[pos];
    //cerr<<model->num_support<<" "<<pos<<" "<<proj<<" "<<alphaKi<<" "<<alpha_old<<" "<<ld->label<<" "<<model->delta[pos]<<" ";

    if(ai > ld->weight)				
      ai = ld->weight;
    else if(ai < 0)
      ai = 0;

    ai *= ld->label;
    double diff = ai - alpha_old;

    if(fabs(diff) > 1.0e-06) 
      overshoot = true;
    
    if(fabs(diff) > 1.) {
      //cerr<<"Here\n";
      diff = sign(diff);
      ai = alpha_old + diff;
    }
    
    for(size_t i = 0;i < model->num_support; i++) {
      label_data* ldi = (label_data*) model->support_vec[i]->ld;
      model->delta[i] += diff*inprods[i]*ldi->label/params->lambda;
    }
    
    //cerr<<model->delta[pos]<<" "<<model->delta[pos]*ai<<" "<<diff<<" ";
    //cerr<<ai<<" "<<diff<<endl;
    //cerr<<"Inprods: ";
    //for(int i = 0;i < model->num_support;i++)
    //cerr<<inprods[i]<<" ";
    //cerr<<endl;
    
    if(fabs(ai) <= 1.0e-10)
      remove(params, pos);
    else {
      model->alpha[pos] = ai;
      //cerr<<ai<<" "<<model->alpha[pos]<<endl;
    }
    

    //double* subopt = new double[model->num_support];
    //size_t max_pos = suboptimality(model, subopt);
    //model->maxdelta = subopt[max_pos];
    //delete[] subopt;
    //delete[] inprods;
    return overshoot;
    //cerr<<model->alpha[pos]<<" "<<subopt[pos]<<endl;
  }

  void train(svm_params* params) {
    
    //cerr<<"In train "<<params->all->training<<endl;
    
    bool* train_pool = (bool*)calloc(params->pool_size, sizeof(bool));
    for(int i = 0;i < params->pool_size;i++)
      train_pool[i] = false;
    
    double* scores = (double*)calloc(params->pool_size, sizeof(double));
    predict(params, params->pool, scores, params->pool_size);
    
      
    if(params->active) {           
      if(params->active_pool_greedy) { 
	multimap<double, int> scoremap;
	for(int i = 0;i < params->pool_size; i++)
	  scoremap.insert(pair<const double, const int>(fabs(scores[i]),i));

	multimap<double, int>::iterator iter = scoremap.begin();
	//cerr<<params->pool_size<<" "<<"Scoremap: ";
	//for(;iter != scoremap.end();iter++)
	//cerr<<iter->first<<" "<<iter->second<<" "<<((label_data*)params->pool[iter->second]->ld)->label<<"\t";
	//cerr<<endl;
	iter = scoremap.begin();
	
	for(size_t train_size = 1;iter != scoremap.end() && train_size <= params->subsample;train_size++) {
	  //cerr<<train_size<<" "<<iter->second<<" "<<iter->first<<endl;
	  train_pool[iter->second] = 1;
	  iter++;	  
	}
      }
      else {

	for(size_t i = 0;i < params->pool_size;i++) {
	  double queryp = 2.0/(1.0 + exp(params->all->active_c0*fabs(scores[i])*pow(params->pool[i]->example_counter,0.5)));
	  if(rand() < queryp) {
	    svm_example* fec = params->pool[i];
	    label_data* ld = (label_data*)fec->ld;
	    ld->weight *= 1/queryp;
	    train_pool[i] = 1;
	  }
	}
      }
      //free(scores);
    }

    if(params->all->training) {
      
      svm_model* model = params->model;
      
      for(size_t i = 0;i < params->pool_size;i++) {
	//cerr<<"process: "<<i<<" "<<train_pool[i]<<endl;;
	int model_pos = -1;
	if(params->active)
	  if(train_pool[i]) {
	    //cerr<<"i = "<<i<<"train_pool[i] = "<<train_pool[i]<<" "<<params->pool[i]->example_counter<<endl;
	    model_pos = add(params, params->pool[i]);
	  }
	  else {
	    delete params->pool[i];
	  }
	else
	  model_pos = add(params, params->pool[i]);
	
	// cerr<<"Added: "<<model->support_vec[model_pos]->example_counter<<endl;
	
	if(model_pos >= 0) {
	  bool overshoot = update(params, model_pos);
	  //cerr<<model_pos<<":alpha = "<<model->alpha[model_pos]<<endl;

	  double* subopt = (double*)calloc(model->num_support,sizeof(double));
	  for(size_t j = 0;j < params->reprocess;j++) {
	    if(model->num_support == 0) break;
	    //cerr<<"reprocess: ";
	    int randi = 1;//rand()%2;
	    if(randi) {
	      size_t max_pos = suboptimality(model, subopt);
	      if(subopt[max_pos] > 0) {
		if(!overshoot && max_pos == model_pos && max_pos > 0 && j == 0) 
		  cerr<<"Shouldn't reprocess right after process!!!\n";
		//cerr<<max_pos<<" "<<subopt[max_pos]<<endl;
		// cerr<<params->model->support_vec[0]->example_counter<<endl;
		if(max_pos*model->num_support <= params->maxcache)
		  make_hot_sv(params, max_pos);
		update(params, max_pos);
	      }
	    }
	    else {
	      size_t rand_pos = rand()%model->num_support;
	      update(params, rand_pos);
	    }
	  }	  
	  //cerr<<endl;
	  // cerr<<params->model->support_vec[0]->example_counter<<endl;
	  free(subopt);
	}
      }

    }
    else
      for(size_t i = 0;i < params->pool_size;i++)
	delete params->pool[i];
	
    // cerr<<params->model->support_vec[0]->example_counter<<endl;
    // for(int i = 0;i < params->pool_size;i++)
    //   cerr<<scores[i]<<" ";
    // cerr<<endl;
    free(scores);
    //cerr<<params->model->support_vec[0]->example_counter<<endl;
    free(train_pool);
    //cerr<<params->model->support_vec[0]->example_counter<<endl;
  }

  void learn(void* d, example* ec) {
    svm_params* params = (svm_params*)d;
    VW::flat_example* fec = VW::flatten_example(*(params->all),ec);

    if(fec) {
      svm_example* sec = new svm_example(fec);
      VW::free_flatten_example(fec);
      double score = 0;
      predict(params, &sec, &score, 1);
      ec->final_prediction = score;
      float label = ((label_data*)ec->ld)->label;
      ec->loss = max(0., 1 - score*label);
      if(params->all->training && ec->example_counter % 100 == 0)
	trim_cache(params);
      if(params->all->training && ec->example_counter % 1000 == 0) {
	cerr<<"Number of support vectors = "<<params->model->num_support<<endl;
	cerr<<"Number of kernel evaluations = "<<num_kernel_evals<<" "<<"Number of cache queries = "<<num_cache_evals<<endl;
      }
      params->pool[params->pool_pos] = sec;
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
	  if(!command_example(all,ec))
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
    for(size_t i = 0;i < model->num_support; i++) {
      delete model->support_vec[i];
      model->support_vec[i] = 0;
     }

    model->support_vec.delete_v();
    model->alpha.delete_v();
    model->delta.delete_v();
    free(model);
  }

  void finish(void* d) {
    svm_params* params = (svm_params*) d;
    free(params->pool);


    cerr<<"Num support = "<<params->model->num_support<<endl;
    cerr<<"Number of kernel evaluations = "<<num_kernel_evals<<" "<<"Number of cache queries = "<<num_cache_evals<<endl;
    //double maxalpha = fabs(params->model->alpha[0]);
    //size_t maxpos = 0;
    
    // for(size_t i = 1;i < params->model->num_support; i++) 
    //   if(maxalpha < fabs(params->model->alpha[i])) {
    // 	maxalpha = fabs(params->model->alpha[i]);
    // 	maxpos = i;
    //   }

    //cerr<<maxalpha<<" "<<maxpos<<endl;

    //cerr<<"Done freeing pool\n";

    free_svm_model(params->model);
    cerr<<"Done freeing model\n";
    if(params->kernel_params) free(params->kernel_params);
    cerr<<"Done freeing kernel params\n";
    free(params);
    cerr<<"Done with finish \n";
  }


  learner setup(vw &all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file) {
    svm_params* params = (svm_params*) calloc(1,sizeof(svm_params));
    cerr<<"In setup\n";

    params->model = (svm_model*) calloc(1,sizeof(svm_model));
    params->model->num_support = 0;
    //params->curcache = 0;
    params->maxcache = 1024*1024*1024;

    //params->model->maxdelta = 0.;
    
    po::options_description desc("KSVM options");
    desc.add_options()
      ("reprocess", po::value<size_t>(), "number of reprocess steps for LASVM")
      ("pool_greedy", "use greedy selection on mini pools")
      ("para_active", "do parallel active learning")
      ("pool_size", po::value<size_t>(), "size of pools for active learning")
      ("subsample", po::value<size_t>(), "number of items to subsample from the pool")
      ("kernel", po::value<string>(), "type of kernel (rbf or linear (default))")
      ("bandwidth", po::value<double>(), "bandwidth of rbf kernel")
      ("lambda", po::value<double>(), "saving regularization for test time");

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
    // cerr<<"Iterating over vm_file of size "<<vm_file.size()<<endl;
    // std::map<std::string, boost::program_options::variable_value>::iterator iter = vm_file.begin();
    // for(;iter != vm_file.end();iter++)
    //   cerr<<(iter->first).c_str()<<endl;

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
    
    params->pool = (svm_example**)calloc(params->pool_size, sizeof(svm_example*));
    params->pool_pos = 0;
    
    if(vm_file.count("subsample"))
      params->subsample = vm["subsample"].as<std::size_t>();
      else if(vm.count("subsample"))
	params->subsample = vm["subsample"].as<std::size_t>();
      else
	params->subsample = 1;
    
    params->lambda = all.l2_lambda;

    std::stringstream ss1, ss2;
    if(!vm_file.count("lambda")) {
      ss1 <<" --lambda "<< params->lambda;
      all.options_from_file.append(ss1.str());
    }
    else {      
      //cerr<<"vm_file[lambda] = ";
      //cerr<<vm_file["lambda"].as<std::string>()<<endl;
      params->lambda = vm_file["lambda"].as<double>();
    }
      
    cerr<<"Lambda = "<<params->lambda<<endl;

    std::string kernel_type;

    if(vm_file.count("kernel") || vm.count("kernel")) {
	
      if(vm_file.count("kernel")) {
	cerr<<"Reading kernel from file\n";
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
      cerr<<"Hopefully saving kernel to model file\n";

      ss2 <<" --kernel "<< kernel_type;
      all.options_from_file.append(ss2.str());
    }

    cerr<<"Kernel = "<<kernel_type<<endl;

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
      cerr<<"bandwidth = "<<bandwidth<<endl;
      params->kernel_params = calloc(1,sizeof(double*));
      *((double*)params->kernel_params) = bandwidth;
      //cerr<<(*(double*)params->kernel_params)<<endl;
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
