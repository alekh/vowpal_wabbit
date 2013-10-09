/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <stdlib.h>
#include <time.h>

#include "constant.h"
#include "oaa.h"
#include "simple_label.h"
#include "cache.h"
#include "v_hashmap.h"
#include "rand48.h"
#include "vw.h"
#include "example.h"
#include "allreduce.h"
#include "accumulate.h"
#include "parser.h"

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
    size_t subsample; 
    example** pool;
    size_t numqueries;
    size_t local_begin, local_end;
    float current_t;
    int save_interval;

    std::string* span_server;
    size_t unique_id; //unique id for each node in the network, id == 0 means extra io.
    size_t total; //total number of nodes
    size_t node; //node id number
    node_socks* socks;
    bool all_done;
    bool local_done;
    time_t start_time;
    
  
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
    //cerr<<"Inpass = "<<n.inpass<<endl;
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
    //cerr<<"Example label = "<<ld->label<<" weight = "<<ld->weight<<endl;
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

    // cerr<<"Weights1: ";
    // for(int i = 0;i <= all.reg.weight_mask;i++)
    //   if(fabs(all.reg.weight_vector[i]) > 0)
    // 	cerr<<i<<":"<<all.reg.weight_vector[i]<<" ";
    // cerr<<endl;

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

    // cerr<<"Weights2: ";
    // for(int i = 0;i <= all.reg.weight_mask;i++)
    //   if(fabs(all.reg.weight_vector[i]) > 0)
    // 	cerr<<i<<":"<<all.reg.weight_vector[i]<<" ";
    // cerr<<endl;

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

    // cerr<<"Weights3: ";
    // for(int i = 0;i <= all.reg.weight_mask;i++)
    //   if(fabs(all.reg.weight_vector[i]) > 0)
    // 	cerr<<i<<":"<<all.reg.weight_vector[i]<<" ";
    // cerr<<endl;


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
      //cerr<<"Before output train "<<((label_data*)ec->ld)->label<<" "<<((label_data*)ec->ld)->weight<<" "<<ec->global_weight<<" "<<ec->example_t<<" "<<ec->eta_round<<endl;
      n.base.learn(&n.output_layer);
      n.output_layer.ld = 0;
    }
    
    // cerr<<"Weights: ";
    // for(int i = 0;i <= all.reg.weight_mask;i++)
    //   if(fabs(all.reg.weight_vector[i]) > 0)
    // 	cerr<<i<<":"<<all.reg.weight_vector[i]<<" ";
    // cerr<<endl;

    n.output_layer.final_prediction = GD::finalize_prediction (all, n.output_layer.partial_prediction);
    //cerr<<"Output final = "<<n.output_layer.final_prediction<<endl;

    if (shouldOutput) {
      outputStringStream << ' ' << n.output_layer.partial_prediction;
      all.print_text(all.raw_prediction, outputStringStream.str(), ec->tag);
    }

    if (all.training && ld->label != FLT_MAX) {
      float gradient = all.loss->first_derivative(all.sd,
                                                  n.output_layer.final_prediction,
                                                  ld->label);
      //cerr<<"gradient = "<<gradient<<endl;

      if (fabs (gradient) > 0) {
        all.loss = n.squared_loss;
        all.set_minmax = noop_mm;
        save_min_label = all.sd->min_label;
        all.sd->min_label = hidden_min_activation;
        save_max_label = all.sd->max_label;
        all.sd->max_label = hidden_max_activation;

	//cerr<<"Gradhw = ";
        for (unsigned int i = 0; i < n.k; ++i) {
          update_example_indicies (all.audit, ec, n.increment);
          if (! dropped_out[i]) {
            float sigmah =
              n.output_layer.atomics[nn_output_namespace][i].x / dropscale;
            float sigmahprime = dropscale * (1.0f - sigmah * sigmah);
            float nu = all.reg.weight_vector[n.output_layer.atomics[nn_output_namespace][i].weight_index & all.reg.weight_mask];
            float gradhw = 0.5f * nu * gradient * sigmahprime;
	    //cerr<<n.increment<<":"<<sigmah<<":"<<nu<<":"<<(n.output_layer.atomics[nn_output_namespace][i].weight_index & all.reg.weight_mask)<<": ";

            ld->label = GD::finalize_prediction (all, hidden_units[i] - gradhw);
            if (ld->label != hidden_units[i])
              n.base.learn(ec);
	    
	    //cerr<<n.increment<<":"<<sigmah<<":"<<nu<<" ";
          }
        }
	//cerr<<endl;
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
    //cerr<<"Final: "<<ec->final_prediction<<endl;
  }


  void learn(void* d,example* ec) {
    nn* n = (nn*)d;
    vw* all = n->all;
    
    learn_with_output(*all, *n, ec, false);
    
  }

  void print_example(vw* all, example* ec) {
    if(command_example(all, ec)) return;
    if(ec->ld) {
      cerr<<"Label: "<<((label_data*)ec->ld)->label<<" "<<((label_data*)ec->ld)->weight<<endl;      
    }
    cerr<<"Prediction: "<<ec->final_prediction<<endl;
    cerr<<"Counter: "<<ec->example_counter<<endl;
    cerr<<"Indices: "<<ec->indices.size()<<" ";
    for(int i = 0;i < ec->indices.size();i++)
      cerr<<(int)ec->indices[i]<<":"<<ec->atomics[(int)ec->indices[i]].size()<<" ";
    cerr<<endl;
    cerr<<"Offset = "<<ec->ft_offset<<endl;
    cerr<<"Num features = "<<ec->num_features<<endl;
    cerr<<"Loss = "<<ec->loss<<endl;
    cerr<<"Eta = "<<ec->eta_round<<" "<<ec->eta_global<<endl;
    cerr<<"Global weight = "<<ec->global_weight<<endl;
    cerr<<"Example t = "<<ec->example_t<<endl;
    cerr<<"Sum_sq = "<<ec->total_sum_feat_sq<<endl;
    cerr<<"Revert weight = "<<ec->revert_weight<<endl;
    cerr<<"Test only = "<<ec->test_only<<endl;
    cerr<<"End pass = "<<ec->end_pass<<endl;
    cerr<<"Sorted = "<<ec->sorted<<endl;
    cerr<<"In use = "<<ec->in_use<<endl;
    cerr<<"Done = "<<ec->done<<endl;
  }


  void sync_queries(vw& all, nn& n, bool* train_pool) {
    io_buf* b = new io_buf();

    char* queries;
    //cerr<<"Syncing"<<endl;
    

    for(int i = 0;i < n.pool_pos;i++) {
      if(!train_pool[i])
	continue;
      //cerr<<n.pool[i]->example_counter<<endl;
      cache_simple_label(n.pool[i]->ld, *b);      
      cache_features(*b, n.pool[i], all.reg.weight_mask);
      //cerr<<"Writing\n";
      //save_load_example(*b, false, n.pool[i]);
      // cerr<<"***********Before**************\n";
      // print_example(&all, n.pool[i]);
    }
    
    float* sizes = (float*)calloc(n.total,sizeof(float));
    sizes[n.node] = b->space.end - b->space.begin;
    //cerr<<"Local size = "<<sizes[all.node]<<endl;
    fflush(stderr);
    all_reduce(sizes, n.total, *n.span_server, n.unique_id, n.total, n.node, *n.socks); 

    //cerr<<"Done with first allreduce\n";
    fflush(stderr);

    //cerr<<"Sizes: ";
    int prev_sum = 0, total_sum = 0;
    for(int i = 0;i < n.total;i++) {
      if(i <= (int)(n.node-1)) {	
    	prev_sum += sizes[i];
      }
      total_sum += sizes[i];
      //cerr<<sizes[i]<<" ";
    }
    //cerr<<endl;
    //cerr<<"Prev sum = "<<prev_sum<<" total_sum = "<<total_sum<<endl;

    if(total_sum > 0) {
      size_t ar_sum = total_sum + (sizeof(float) - total_sum % sizeof(float)) % sizeof(float);
      queries = (char*)calloc(ar_sum, sizeof(char));
      memset(queries, '\0', ar_sum);
      memcpy(queries + prev_sum, b->space.begin, b->space.end - b->space.begin);
      //cerr<<"Copied "<<(b->space.end - b->space.begin)<<endl;
      b->space.delete_v();
      //cerr<<"Entering second allreduce\n";
      fflush(stderr);
      all_reduce(queries, ar_sum, *n.span_server, n.unique_id, n.total, n.node, *n.socks); 

      //cerr<<"Done with second allreduce\n";
      fflush(stderr);
      
      b->space.begin = queries;
      b->space.end = b->space.begin;
      b->endloaded = &queries[total_sum*sizeof(char)];

      //cerr<<"Before reading: "<<b->endloaded - b->space.begin<<" "<<b->space.end - b->space.begin<<" "<<b->endloaded - b->space.end<<endl;

      size_t num_read = 0;
      n.pool_pos = 0;
      float label_avg = 0, weight_sum = 0;
      float min_weight = FLT_MAX, max_weight = -1;
      int min_pos = -1, max_pos = -1;
      for(size_t i = 0;num_read < total_sum; n.pool_pos++,i++) {            
    	n.pool[i] = (example*) calloc(1, sizeof(example));
    	n.pool[i]->ld = calloc(1, sizeof(simple_label));
	//cerr<<"i = "<<i<<" "<<num_read<<endl;
    	if(read_cached_features(&all, *b, n.pool[i])) {
	  //if(!save_load_example(*b, true, n.pool[i])) {
	  //cerr<<"***********After**************\n";
    	  train_pool[i] = true;
    	  n.pool[i]->in_use = true;	
	  float weight = ((label_data*) n.pool[i]->ld)->weight;
	  n.current_t += weight;
	  //cerr<<"Current_t = "<<n.current_t<<endl;
	  n.pool[i]->example_t = n.current_t;	  
	  label_avg += weight * ((label_data*) n.pool[i]->ld)->label;
	  weight_sum += weight;
	  if(weight > max_weight) {
	    max_weight = weight;
	    max_pos = i;
	  }
	  if(weight < min_weight) {
	    min_weight = weight;
	    min_pos = i;
	  }
    	  // print_example(&all, n.pool[i]);
    	}
    	else
    	  break;
	//cerr<<b->endloaded - b->space.begin<<" "<<b->space.end - b->space.begin<<" "<<b->endloaded - b->space.end<<endl;
	
    	num_read = min(b->space.end - b->space.begin,b->endloaded - b->space.begin);
    	if(num_read == prev_sum)
    	  n.local_begin = i+1;
    	if(num_read == prev_sum + sizes[n.node])
    	  n.local_end = i;
	//cerr<<"num_read = "<<num_read<<endl;
      }
      //cerr<<"Sum of labels = "<<label_avg<<" average weight= "<<weight_sum/n.pool_pos<<" average = "<<label_avg/weight_sum<<" "<<min_weight<<" "<<min_pos<<" "<<max_weight<<" "<<max_pos<<endl;
      
    }

    //cerr<<"Synced\n";
    free(sizes);
    delete b;
  }


  void predict_and_learn(vw& all, nn& n,  example** ec_arr, bool shouldOutput) {
    
    float* gradients = (float*)calloc(n.pool_pos, sizeof(float));
    bool* train_pool = (bool*)calloc(n.pool_size*n.pool_size, sizeof(bool));
    size_t* local_pos = (size_t*) calloc(n.pool_pos, sizeof(size_t));
    //cerr<<"Predicting\n";
    if(n.active) {
      float gradsum = 0;
      for(int idx = 0;idx < n.pool_pos;idx++) {
	example* ec = n.pool[idx];
	train_pool[idx] = false;
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

      if(n.para_active)
	all_reduce(&gradsum, 1, *n.span_server, n.unique_id, n.total, n.node, *n.socks);
      
      multimap<float, int, std::greater<float> > scoremap;
      for(int i = 0;i < (&n)->pool_pos; i++)
	scoremap.insert(pair<const float, const int>(gradients[i],i));
      
      multimap<float, int, std::greater<float> >::iterator iter = scoremap.begin();
      float* queryp = (float*)calloc(n.pool_pos, sizeof(float));
      float querysum = 0;
      
      for(int i = 0;i < n.pool_pos;i++) {
	queryp[i] = min<float>(gradients[i]/gradsum*(float)n.subsample, 1.0);
	//cerr<<queryp[i]<<":"<<gradients[i]/gradsum * (float)n.subsample<<" ";
	querysum += queryp[i];
      }

      float residual = n.subsample - querysum;
      
      for(int pos = 0;iter != scoremap.end() && residual > 0;iter++, pos++) {
	if(pos == n.pool_pos)
	  cerr<<"Problem: n.pool_pos == pos\n";
	if(queryp[iter->second] + residual/(n.pool_pos - pos) <= 1) {
	  queryp[iter->second] += residual/(n.pool_pos - pos);
	  residual -= residual/(n.pool_pos - pos);
	}
	else {
	  residual -= (1 - queryp[iter->second]);
	  queryp[iter->second] = 1;
	}
	
      }

      int num_train = 0;
      float label_avg = 0, weight_sum = 0;

      for(int i = 0;i < n.pool_pos && num_train < n.subsample + 1;i++)
	//for(int i = 0;i < n.pool_pos;i++)
	if(frand48() < queryp[i]) {
	  train_pool[i] = true;
	  label_data* ld = (label_data*) n.pool[i]->ld;
	  ld->weight = 1/queryp[i]/n.pool_size;
	  local_pos[num_train] = i;
	  n.numqueries++;
	  num_train++;
	  label_avg += ((label_data*) n.pool[i]->ld)->weight * ((label_data*) n.pool[i]->ld)->label;
	  weight_sum += ((label_data*) n.pool[i]->ld)->weight;
	}
      
      //if(weight_sum > 0)
	//cerr<<"Sum of labels = "<<label_avg<<" weight_sum = "<<weight_sum<<" average = "<<label_avg/weight_sum<<endl;

      // for(int i = 0; i < n.pool_pos;i++) 
      // 	cerr<<"gradient: "<<gradients[i]<<" queryp: "<<queryp[i]<<" ";
      // cerr<<endl;

      free(queryp);
      //cerr<<"Locally selecting "<<num_train<<" "<<scoremap.begin()->first<<" "<<gradsum<<endl;
    }
    
    //cerr<<"Calling sync\n";
    
    
    if(n.para_active) 
      sync_queries(all, n, train_pool);

    //cerr<<"Globally collected "<<n.pool_pos<<endl;

    for(int i = 0;i < n.pool_pos;i++) {
      if(n.active && !train_pool[i])
	continue;
      
      example* ec = n.pool[i];
      
      learn_with_output(all, n, ec, shouldOutput);
      
      if(n.para_active) {
	if(i >= n.local_begin && i<= n.local_end) {	  
	  int pos = local_pos[i-n.local_begin];
	  ec_arr[pos]->final_prediction = n.pool[i]->final_prediction;
	  ec_arr[pos]->loss = n.pool[i]->loss;
	}
	dealloc_example(NULL, *(n.pool[i]));
	free(n.pool[i]);
      }
    }
    
    free(local_pos);
    free(train_pool);
    free(gradients);
  }

  void drive_nn(vw *all, void* d)
  {
    //cerr<<"In driver\n";
    fflush(stderr);
    nn* n = (nn*)d;
    example** ec_arr = (example**)calloc(n->pool_size, sizeof(example*));
    for(int i = 0;i < n->pool_size;i++)
      ec_arr[i] = NULL;
    int local_pos = 0;
    bool command = false;
    int num_read = 0;
    
    // int one = 1;
    // all_reduce(&one, 1, *n->span_server, n->unique_id, n->total, n->node, *n->socks);
    // cerr<<"After first AR "<<one<<endl;

    while ( true )
      {
	for(int i = 0;i < n->pool_size;i++) {
	  if(n->local_done) break;
	  //cerr<<i<<" "<<n->pool_size<<" "<<n->pool_pos<<endl;
	  //fflush(stderr);
	  if ((ec_arr[i] = VW::get_example(all->p)) != NULL)//semiblocking operation.
	    {
	      local_pos++;
	      //cerr<<"Read new example\n";
	      fflush(stderr);
	      if(!command_example(all,ec_arr[i])) {
		//cerr<<"Putting in the pool\n";
		if(num_read % n->save_interval == 1) {
		  time_t now;
		  double elapsed;
		  time(&now);
		  elapsed = difftime(now, n->start_time);
		  
		  string final_regressor_name = all->final_regressor_name;
		  int model_num = num_read / n->save_interval;
		  char buffer[50];
		  sprintf(buffer, ".%d", model_num);
		  final_regressor_name.append(buffer);
		  cerr<<"Saving model to "<<final_regressor_name<<" time elapsed = "<<elapsed<<endl;
		  save_predictor(*all, final_regressor_name, 0);
		}

		n->pool[n->pool_pos++] = ec_arr[i];		
		//cerr<<ec_arr[i]->in_use<<" "<<n->pool[i]->in_use<<" "<<command_example(all,ec_arr[i])<<" "<<n->pool_pos<<" "<<ec_arr[i]->example_counter<<endl;
		num_read++;
	      }
	      else {
		cerr<<"Found command example!!!\n";
		command = true;
	      }
	    }
	  else {
	    //cerr<<"Parser says NULL\n";
	    break;
	  }
	}
	// if(command) 
	//   cerr<<"pool_pos = "<<n->pool_pos<<endl;
	// //local_pos = n->pool_pos;
	// cerr<<"Calling predict and learn, local_pos = "<<local_pos<<endl;
	
	predict_and_learn(*all, *n, ec_arr, false);
	// if(command) 
	//   cerr<<"After predict and learn, pool_pos = "<<n->pool_pos<<endl;
	for(int i = 0;i < local_pos;i++) {
	  // float save_label = ((label_data*)ec_arr[i]->ld)->label;
	  // ((label_data*)ec_arr[i]->ld)->label = FLT_MAX;
	  // n->base.learn(ec_arr[i]);
	  // ((label_data*)ec_arr[i]->ld)->label = save_label;
	  // ec_arr[i]->loss = all->loss->getLoss(all->sd, ec_arr[i]->final_prediction, save_label);
	  //cerr<<"While accounting: ";
	  //cerr<<((label_data*)ec_arr[i]->ld)->label<<" "<<ec_arr[i]->loss<<" "<<ec_arr[i]->final_prediction<<endl;
	  int save_raw_prediction = all->raw_prediction;
	  all->raw_prediction = -1;
	  return_simple_example(*all, ec_arr[i]);
	  all->raw_prediction = save_raw_prediction;			  
	}
	
	int done = (int)parser_done(all->p);
	//cerr<<"Done = "<<done<<endl;
	if(done) n->local_done = true;
	if(n->para_active) {
	  all_reduce(&done, 1, *n->span_server, n->unique_id, n->total, n->node, *(n->socks)); 
	  // if(command)
	  //   cerr<<"All done = "<<done<<endl;
	  if(done > 0) {
	    cerr<<n->pool_pos<<" "<<done<<endl;
	  }
	  if(done == n->total)
	    n->all_done = true;
	}

	if((!n->para_active && n->local_done) || (n->para_active && n->all_done)) {
	  free(ec_arr);
	  cerr<<"Parser done \n";
	   if(n->para_active) {
	     cerr<<"Aggregating things at the end\n";
	     cerr<<n->socks->parent<<" "<<n->socks->children[0]<<" "<<n->socks->children[1]<<endl;
	     all_reduce(&all->sd->example_number, 1, *n->span_server, n->unique_id, n->total, n->node, *n->socks);
	     all_reduce(&all->sd->weighted_examples, 1, *n->span_server, n->unique_id, n->total, n->node, *n->socks);
	     all_reduce(&all->sd->sum_loss, 1, *n->span_server, n->unique_id, n->total, n->node, *n->socks);
	     all_reduce(&n->numqueries, 1, *n->span_server, n->unique_id, n->total, n->node, *n->socks);
	     cerr<<"Loss "<<all->sd->sum_loss<<" "<<all->sd->weighted_examples<<endl;
	   }
	   return;
	}
	n->pool_pos = 0;
	local_pos = 0;
      }

    
  }

  void finish(void* d)
  {
    nn* n =(nn*)d;
    vw* all = n->all;
    if(n->active)
      cerr<<"Number of label queries = "<<n->numqueries<<endl;   
    time_t now;
    double elapsed;
    time(&now);
    elapsed = difftime(now, n->start_time);
    cerr<<"Total time elapsed = "<<elapsed<<endl;
    n->base.finish();
    delete n->squared_loss;
    free (n->output_layer.indices.begin);
    free (n->output_layer.atomics[nn_output_namespace].begin);
    free(n->pool);
    delete n->socks;
    delete n->span_server;
    free(n);
  }

  learner setup(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file)
  {    
    nn* n = (nn*)calloc(1,sizeof(nn));
    n->all = &all;
    time(&n->start_time);
    cerr<<"Size = "<<sizeof(nn)<<" "<<sizeof(std::string)<<" "<<sizeof(n->span_server)<<endl;

    po::options_description desc("NN options");
    desc.add_options()
      ("pool_greedy", "use greedy selection on mini pools")
      ("para_active", "do parallel active learning")
      ("pool_size", po::value<size_t>(), "size of pools for active learning")
      ("subsample", po::value<size_t>(), "number of items to subsample from the pool")
      ("inpass", "Train or test sigmoidal feedforward network with input passthrough.")
      ("dropout", "Train or test sigmoidal feedforward network using dropout.")
      ("meanfield", "Train or test sigmoidal feedforward network using mean field.")
      ("save_interval", po::value<int>(), "Number of examples before saving the model");

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
    cerr<<"Node = "<<(uint64_t) all.node<<endl;
    msrand48((uint64_t)all.node);

    n->training = all.training;
    cerr<<"training = "<<n->training<<endl;
    n->active = all.active_simulation;        
    if(n->active) {
      if(vm.count("pool_greedy"))
	n->active_pool_greedy = 1;
      if(vm.count("para_active"))
	n->para_active = 1;
      n->numqueries = 0;
      //cerr<<"C0 = "<<all.active_c0<<endl;
      if(n->para_active)
	n->current_t = 0;
      all.l.finish();
    }
    
    cerr<<"Finished base learner\n";
    if(n->para_active) {
      n->span_server = new std::string(all.span_server);
      //>span_server.assign(all.span_server.c_str());
      n->total = all.total;
      n->unique_id = all.unique_id;
      n->node = all.node;
      n->socks = new node_socks();
      n->socks->current_master = all.socks.current_master;
      // if(all.socks.parent)
      // 	n->socks->parent = all.socks.parent;
      // if(all.socks.children) {
      // 	n->socks->children[0] = all.socks.children[0];
      // 	n->socks->children[1] = all.socks.children[1];
      //}
      all.span_server = "";
      all.total = 0;      
      n->all_done = false;
      n->local_done = false;
      //delete &all.socks;
    }
    cerr<<"Copied fields from all\n";
    //cerr<<*n->span_server<<" "<<n->total<<" "<<n->unique_id<<" "<<n->node<<" "<<n->socks->current_master<<endl; 
	
    all.active_simulation = false;   
    cerr<<"Calling setup again\n";
    all.l = GD::setup(all, vm);
    
    if(vm_file.count("pool_size"))
      n->pool_size = vm_file["pool_size"].as<std::size_t>();
    else if(vm.count("pool_size")) 
      n->pool_size = vm["pool_size"].as<std::size_t>();
    else
      n->pool_size = 1;

    if(vm_file.count("save_interval"))
      n->save_interval = vm_file["save_interval"].as<int>();
    else if(vm.count("save_interval")) 
      n->save_interval = vm["save_interval"].as<int>();
    else
      n->save_interval = -1;
    
    n->pool = (example**)calloc(n->pool_size*n->pool_size, sizeof(example*));
    n->pool_pos = 0;
    
    if(vm_file.count("subsample"))
      n->subsample = vm["subsample"].as<std::size_t>();
      else if(vm.count("subsample"))
	n->subsample = vm["subsample"].as<std::size_t>();
      else if(n->para_active)
	n->subsample = ceil(n->pool_size / n->total);
      else
	n->subsample = 1;
    cerr<<"Subsample = "<<n->subsample<<endl;

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
    else
      n->dropout = false;

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
