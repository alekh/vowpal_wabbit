/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>

#include "constant.h"
#include "oaa.h"
#include "simple_label.h"
#include "cache.h"
#include "v_hashmap.h"
#include "rand48.h"
#include "vw.h"
#include "example.h"

using namespace std;

namespace NN {
  const float hidden_min_activation = -3;
  const float hidden_max_activation = 3;
  const int nn_constant = 533357803;
  
  struct nn {
    uint32_t k;
    uint32_t increment;
    loss_function* squared_loss;
    example output_layer;
    bool dropout;
    uint64_t xsubi;
    uint64_t save_xsubi;
    bool inpass;
    bool finished_setup;

    //active flags
    bool active;
    bool active_pool_greedy;
    bool para_active;
    bool training;

    //pool maintainence
    size_t pool_size;
    size_t pool_pos;
    size_t subsample; //NOTE: Eliminating subsample to only support 1/pool_size
    example** pool;
    size_t numqueries;

  
    learner base;
    vw* all;
  };

#define cast_uint32_t static_cast<uint32_t>

  static inline float
  fastpow2 (float p)
  {
    float offset = (p < 0) ? 1.0f : 0.0f;
    float clipp = (p < -126) ? -126.0f : p;
    int w = (int)clipp;
    float z = clipp - w + offset;
    union { uint32_t i; float f; } v = { cast_uint32_t ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

    return v.f;
  }

  static inline float
  fastexp (float p)
  {
    return fastpow2 (1.442695040f * p);
  }

  static inline float
  fasttanh (float p)
  {
    return -1.0f + 2.0f / (1.0f + fastexp (-2.0f * p));
  }

  static void
  update_atomics_indices (v_array<feature>& f,
                          uint32_t          offset)
    {
      for (feature* x = f.begin; x != f.end; ++x)
        {
          x->weight_index += offset;
        }
    }

  void finish_setup (nn* n, vw& all);


  void learn_with_output(vw& all, nn& n, example* ec, bool shouldOutput)
  {
    if (! n.finished_setup)
      finish_setup (&n, all);

    if (ec->end_pass) {
      if (all.bfgs)
        n.xsubi = n.save_xsubi;
    }

    if (command_example(&all, ec)) {
      n.base.learn(ec);
      return;
    }

    label_data* ld = (label_data*)ec->ld;
    float save_label = ld->label;
    void (*save_set_minmax) (shared_data*, float) = all.set_minmax;
    float save_min_label;
    float save_max_label;
    float dropscale = n.dropout ? 2.0f : 1.0f;
    loss_function* save_loss = all.loss;

    float* hidden_units = (float*) alloca (n.k * sizeof (float));
    bool* dropped_out = (bool*) alloca (n.k * sizeof (bool));
  
    string outputString;
    stringstream outputStringStream(outputString);

    all.set_minmax = noop_mm;
    all.loss = n.squared_loss;
    save_min_label = all.sd->min_label;
    all.sd->min_label = hidden_min_activation;
    save_max_label = all.sd->max_label;
    all.sd->max_label = hidden_max_activation;
    ld->label = FLT_MAX;
    for (unsigned int i = 0; i < n.k; ++i)
      {
        update_example_indicies(all.audit, ec, n.increment);

        n.base.learn(ec);
        hidden_units[i] = ec->final_prediction;

        dropped_out[i] = (n.dropout && merand48 (n.xsubi) < 0.5);

        if (shouldOutput) {
          if (i > 0) outputStringStream << ' ';
          outputStringStream << i << ':' << ec->partial_prediction << ',' << fasttanh (hidden_units[i]);
        }
      }
    update_example_indicies(all.audit, ec, -n.k * n.increment);
    ld->label = save_label;
    all.loss = save_loss;
    all.set_minmax = save_set_minmax;
    all.sd->min_label = save_min_label;
    all.sd->max_label = save_max_label;

    bool converse = false;
    float save_partial_prediction = 0;
    float save_final_prediction = 0;
    float save_ec_loss = 0;

CONVERSE: // That's right, I'm using goto. So sue me.

    n.output_layer.total_sum_feat_sq = 1;
    n.output_layer.sum_feat_sq[nn_output_namespace] = 1;

    for (unsigned int i = 0; i < n.k; ++i)
      {
        float sigmah =
          (dropped_out[i]) ? 0.0f : dropscale * fasttanh (hidden_units[i]);
        n.output_layer.atomics[nn_output_namespace][i].x = sigmah;

        n.output_layer.total_sum_feat_sq += sigmah * sigmah;
        n.output_layer.sum_feat_sq[nn_output_namespace] += sigmah * sigmah;
      }

    if (n.inpass) {
      // TODO: this is not correct if there is something in the
      // nn_output_namespace but at least it will not leak memory
      // in that case

      update_atomics_indices (n.output_layer.atomics[nn_output_namespace], -ec->ft_offset);
      ec->indices.push_back (nn_output_namespace);
      v_array<feature> save_nn_output_namespace = ec->atomics[nn_output_namespace];
      ec->atomics[nn_output_namespace] = n.output_layer.atomics[nn_output_namespace];
      ec->sum_feat_sq[nn_output_namespace] = n.output_layer.sum_feat_sq[nn_output_namespace];
      ec->total_sum_feat_sq += n.output_layer.sum_feat_sq[nn_output_namespace];
      n.base.learn(ec);
      n.output_layer.partial_prediction = ec->partial_prediction;
      n.output_layer.loss = ec->loss;
      ec->total_sum_feat_sq -= n.output_layer.sum_feat_sq[nn_output_namespace];
      ec->sum_feat_sq[nn_output_namespace] = 0;
      ec->atomics[nn_output_namespace] = save_nn_output_namespace;
      ec->indices.pop ();
      update_atomics_indices (n.output_layer.atomics[nn_output_namespace], ec->ft_offset);
    }
    else {
      n.output_layer.ld = ec->ld;
      n.output_layer.partial_prediction = 0;
      n.output_layer.eta_round = ec->eta_round;
      n.output_layer.eta_global = ec->eta_global;
      n.output_layer.global_weight = ec->global_weight;
      n.output_layer.example_t = ec->example_t;
      n.base.learn(&n.output_layer);
      n.output_layer.ld = 0;
    }

    n.output_layer.final_prediction = GD::finalize_prediction (all, n.output_layer.partial_prediction);

    if (shouldOutput) {
      outputStringStream << ' ' << n.output_layer.partial_prediction;
      all.print_text(all.raw_prediction, outputStringStream.str(), ec->tag);
    }

    if (all.training && ld->label != FLT_MAX) {
      float gradient = all.loss->first_derivative(all.sd,
                                                  n.output_layer.final_prediction,
                                                  ld->label);

      if (fabs (gradient) > 0) {
        all.loss = n.squared_loss;
        all.set_minmax = noop_mm;
        save_min_label = all.sd->min_label;
        all.sd->min_label = hidden_min_activation;
        save_max_label = all.sd->max_label;
        all.sd->max_label = hidden_max_activation;

        for (unsigned int i = 0; i < n.k; ++i) {
          update_example_indicies (all.audit, ec, n.increment);
          if (! dropped_out[i]) {
            float sigmah =
              n.output_layer.atomics[nn_output_namespace][i].x / dropscale;
            float sigmahprime = dropscale * (1.0f - sigmah * sigmah);
            float nu = all.reg.weight_vector[n.output_layer.atomics[nn_output_namespace][i].weight_index & all.reg.weight_mask];
            float gradhw = 0.5f * nu * gradient * sigmahprime;

            ld->label = GD::finalize_prediction (all, hidden_units[i] - gradhw);
            if (ld->label != hidden_units[i])
              n.base.learn(ec);
          }
        }
        update_example_indicies (all.audit, ec, -n.k*n.increment);

        all.loss = save_loss;
        all.set_minmax = save_set_minmax;
        all.sd->min_label = save_min_label;
        all.sd->max_label = save_max_label;
      }
    }

    ld->label = save_label;

    if (! converse) {
      save_partial_prediction = n.output_layer.partial_prediction;
      save_final_prediction = n.output_layer.final_prediction;
      save_ec_loss = n.output_layer.loss;
    }

    if (n.dropout && ! converse)
      {
        for (unsigned int i = 0; i < n.k; ++i)
          {
            dropped_out[i] = ! dropped_out[i];
          }

        converse = true;
        goto CONVERSE;
      }

    ec->partial_prediction = save_partial_prediction;
    ec->final_prediction = save_final_prediction;
    ec->loss = save_ec_loss;
  }
  void learn(void* d,example* ec) {
    nn* n = (nn*)d;
    vw* all = n->all;
    
    learn_with_output(*all, *n, ec, false);
    
  }


  void sync_queries(vw& all, nn& n, bool* train_pool) {
    io_buf b;
    char* queries;
    
    for(int i = 0;i < n.pool_pos;i++) {
      if(!train_pool[i])
	continue;
      //b.init();
      
      cache_simple_label(n.pool[i]->ld, b);      
      cache_features(b, n.pool[i], mask);
    }
    
    float* sizes = (float*)calloc(all.total,sizeof(float));
    sizes[all.unique_id] = b.space.size();
    allreduce(sizes, all.total, all.master_location, all.unique_id, all.total, all.node, all.socks); 
    
    int prev_sum = 0, total_sum = 0;
    for(size_t i = 0;i < all.total;i++) {
      if(i < unique_id-1)	
	prev_sum += sizes[i];
      total_sum += sizes[i];
    }
    
    queries = (char*)calloc(total_sum, sizeof(char));
    memcpy(queries + prev_sum, b.space.begin, b.space.size());
    allreduce(queries, total_sum, all.master_location, all.unique_id, all.total, all.node, all.socks); 

    io_buf* save_in = all.p->in;
    //size_t read_sum = 0;
    all.p->in = b;
    b.space.begin = queries;
    b.space.end = b.space.begin;
    b.endloaded = queries.end;
    for(size_t i = 0;i < n.pool_size, n.pool_pos++;i++) {      
      n.pool[i] = (example*) calloc(1, sizeof(example));
      if(read_cached_features(all, n.pool[i]))
	train_pool[i] = true;
      else
	break;
    }
  }


  void predict_and_learn(vw& all, nn& n,  bool shouldOutput) {
    
    float* gradients = (float*)calloc(n.pool_pos, sizeof(float));
    bool* train_pool = (bool*)calloc(n.pool_pos, sizeof(bool));

    if(n.active) {
      float gradsum = 0;
      for(int idx = 0;idx < n.pool_pos;idx++) {
	example* ec = n.pool[idx];
	
	if (! n.finished_setup)
	  finish_setup (&n, all);
	
	if (command_example(&all, ec)) {
	  n.base.learn(ec);
	  break;
	}
	
	label_data* ld = (label_data*) ec->ld;
	float save_label = ld->label;
	ld->label = FLT_MAX;
	
	learn_with_output(all, n, ec, shouldOutput);
	ld->label = save_label;
	gradients[idx] = fabs(all.loss->first_derivative(all.sd, ec->final_prediction, ld->label));      
	gradsum += gradients[idx];
	ec->loss = all.loss->getLoss(all.sd, ec->final_prediction, ld->label);
      }
      
      multimap<float, int, std::greater<float> > scoremap;
      for(int i = 0;i < (&n)->pool_pos; i++)
	scoremap.insert(pair<const float, const int>(gradients[i],i));
      
      multimap<float, int, std::greater<float> >::iterator iter = scoremap.begin();
      float* queryp = (float*)calloc(n.pool_pos, sizeof(float));
      float querysum = 0;
      
      for(int i = 0;i < n.pool_pos;i++) {
	queryp[i] = min<float>(gradients[i]/gradsum * (float)n.subsample, 1.0);
	//cerr<<queryp[i]<<":"<<gradients[i]/gradsum * (float)n.subsample<<" ";
	querysum += queryp[i];
      }

      float residual = n.subsample - querysum;
      
      for(;iter != scoremap.end() && residual > 0;iter++) {
	if(queryp[iter->second] + residual <= 1) {
	  queryp[iter->second] += residual;
	  residual = 0;
	}
	else {
	  residual -= (1 - queryp[iter->second]);
	  queryp[iter->second] = 1;
	}
	
      }

      int num_train = 0;

      for(int i = 0;i < n.pool_pos && num_train < n.subsample;i++)
	if(frand48() < queryp[i]) {
	  train_pool[i] = 1;
	  label_data* ld = (label_data*) n.pool[i]->ld;
	  ld->weight = 1/queryp[i];
	  n.numqueries++;
	  num_train++;
	}

      // for(int i = 0; i < n.pool_pos;i++) 
      // 	cerr<<"gradient: "<<gradients[i]<<" queryp: "<<queryp[i]<<" ";
      // cerr<<endl;

      free(queryp);
    }

    if(n.para_active) 
      sync_queries(all, n, train_pool);

    for(int i = 0;i < n.pool_pos;i++) {
      if(n.active && !train_pool[i])
	continue;
      
      example* ec = n.pool[i];
      learn_with_output(all, n, ec, shouldOutput);
      if(n.para_active)
	return_simple_example(*all, n.pool[i]);
      
    }
    
    free(train_pool);
    free(gradients);
  }

  void drive_nn(vw *all, void* d)
  {
    nn* n = (nn*)d;
    example** ec_arr = (example**)calloc(n->pool_size, sizeof(example*));
    for(int i = 0;i < n->pool_size;i++)
      ec_arr[i] = NULL;
    
    while ( true )
      {
	for(int i = 0;i < n->pool_size;i++) {
	  if ((ec_arr[i] = VW::get_example(all->p)) != NULL)//semiblocking operation.
	    {
	      n->pool[i] = ec_arr[i];
	      n->pool_pos++;
	    }
	  else 
	    break;
	}
	
	predict_and_learn(*all, *n, false);
	for(int i = 0;i < n->pool_pos;i++) {
	  int save_raw_prediction = all->raw_prediction;
	  all->raw_prediction = -1;
	  return_simple_example(*all, ec_arr[i]);
	  all->raw_prediction = save_raw_prediction;			  
	}
	n->pool_pos = 0;
	if(parser_done(all->p))
	   return;
      }
  }

  void finish(void* d)
  {
    nn* n =(nn*)d;
    if(n->active)
      cerr<<"Number of label queries = "<<n->numqueries<<endl;
    n->base.finish();
    delete n->squared_loss;
    free (n->output_layer.indices.begin);
    free (n->output_layer.atomics[nn_output_namespace].begin);
    free(n);
  }

  learner setup(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file)
  {
    nn* n = (nn*)calloc(1,sizeof(nn));
    n->all = &all;

    po::options_description desc("NN options");
    desc.add_options()
      ("pool_greedy", "use greedy selection on mini pools")
      ("para_active", "do parallel active learning")
      ("pool_size", po::value<size_t>(), "size of pools for active learning")
      ("subsample", po::value<size_t>(), "number of items to subsample from the pool")
      ("inpass", "Train or test sigmoidal feedforward network with input passthrough.")
      ("dropout", "Train or test sigmoidal feedforward network using dropout.")
      ("meanfield", "Train or test sigmoidal feedforward network using mean field.");

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

    n->training = all.training;
    n->active = all.active_simulation;        
    all.active_simulation = false;
    all.l.finish();
    all.l = GD::setup(all, vm);
    if(n->active) {
      if(vm.count("pool_greedy"))
	n->active_pool_greedy = 1;
      if(vm.count("para_active"))
	n->para_active = 1;
      n->numqueries = 0;
      cerr<<"C0 = "<<all.active_c0<<endl;
    }
    
    if(vm_file.count("pool_size"))
      n->pool_size = vm_file["pool_size"].as<std::size_t>();
    else if(vm.count("pool_size")) 
      n->pool_size = vm["pool_size"].as<std::size_t>();
    else
      n->pool_size = 1;
    
    n->pool = (example**)calloc(n->pool_size, sizeof(example*));
    n->pool_pos = 0;
    
    if(vm_file.count("subsample"))
      n->subsample = vm["subsample"].as<std::size_t>();
      else if(vm.count("subsample"))
	n->subsample = vm["subsample"].as<std::size_t>();
      else if(n->para_active)
	n->subsample = ceil(n->pool_size / all.total);
      else
	n->subsample = 1;

    //first parse for number of hidden units
    n->k = 0;
    if( vm_file.count("nn") ) {
      n->k = (uint32_t)vm_file["nn"].as<size_t>();
      if( vm.count("nn") && (uint32_t)vm["nn"].as<size_t>() != n->k )
        std::cerr << "warning: you specified a different number of hidden units through --nn than the one loaded from predictor. Pursuing with loaded value of: " << n->k << endl;
    }
    else {
      n->k = (uint32_t)vm["nn"].as<size_t>();

      std::stringstream ss;
      ss << " --nn " << n->k;
      all.options_from_file.append(ss.str());
    }

    if( vm_file.count("dropout") ) {
      n->dropout = all.training || vm.count("dropout");

      if (! n->dropout && ! vm.count("meanfield") && ! all.quiet) 
        std::cerr << "using mean field for testing, specify --dropout explicitly to override" << std::endl;
    }
    else if ( vm.count("dropout") ) {
      n->dropout = true;

      std::stringstream ss;
      ss << " --dropout ";
      all.options_from_file.append(ss.str());
    }

    if ( vm.count("meanfield") ) {
      n->dropout = false;
      if (! all.quiet) 
        std::cerr << "using mean field for neural network " 
                  << (all.training ? "training" : "testing") 
                  << std::endl;
    }

    if (n->dropout) 
      if (! all.quiet)
        std::cerr << "using dropout for neural network "
                  << (all.training ? "training" : "testing") 
                  << std::endl;

    if( vm_file.count("inpass") ) {
      n->inpass = true;
    }
    else if (vm.count ("inpass")) {
      n->inpass = true;

      std::stringstream ss;
      ss << " --inpass";
      all.options_from_file.append(ss.str());
    }

    if (n->inpass && ! all.quiet)
      std::cerr << "using input passthrough for neural network "
                << (all.training ? "training" : "testing") 
                << std::endl;

    n->base = all.l;

    n->increment = all.reg.stride * all.weights_per_problem;
    all.weights_per_problem *= n->k + 1;

    n->finished_setup = false;
    n->squared_loss = getLossFunction (0, "squared", 0);

    n->xsubi = 0;

    if (vm.count("random_seed"))
      n->xsubi = vm["random_seed"].as<size_t>();

    n->save_xsubi = n->xsubi;

    learner l(n,drive_nn,learn,finish,all.l.sl);
    return l;
  }

  void finish_setup (nn* n, vw& all)
  {
    bool initialize = true;

    // TODO: output_layer audit

    memset (&n->output_layer, 0, sizeof (n->output_layer));
    n->output_layer.indices.push_back(nn_output_namespace);
    feature output = {1., nn_constant*all.reg.stride};

    for (unsigned int i = 0; i < n->k; ++i)
      {
        n->output_layer.atomics[nn_output_namespace].push_back(output);
        initialize &= (all.reg.weight_vector[output.weight_index & all.reg.weight_mask] == 0);
        ++n->output_layer.num_features;
        output.weight_index += n->increment;
      }

    if (! n->inpass) 
      {
        n->output_layer.atomics[nn_output_namespace].push_back(output);
        initialize &= (all.reg.weight_vector[output.weight_index & all.reg.weight_mask] == 0);
        ++n->output_layer.num_features;
      }

    n->output_layer.in_use = true;

    if (initialize) {
      // output weights

      float sqrtk = sqrt ((float)n->k);
      for (feature* x = n->output_layer.atomics[nn_output_namespace].begin; 
           x != n->output_layer.atomics[nn_output_namespace].end; 
           ++x)
        {
          weight* w = &all.reg.weight_vector[x->weight_index & all.reg.weight_mask];

          w[0] = (float) (frand48 () - 0.5) / sqrtk;

          // prevent divide by zero error
          if (n->dropout && all.normalized_updates)
            w[all.normalized_idx] = 1e-4f;
        }

      // hidden biases

      unsigned int weight_index = constant * all.reg.stride;

      for (unsigned int i = 0; i < n->k; ++i)
        {
          weight_index += n->increment;
          all.reg.weight_vector[weight_index & all.reg.weight_mask] = (float) (frand48 () - 0.5);
        }
    }

    n->finished_setup = true;
  }
}
