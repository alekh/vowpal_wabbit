/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#ifdef _WIN32
#include <WinSock2.h>
#else
#include <netdb.h>
#endif

#define SVM_KER_LIN 0
#define SVM_KER_RBF 1
#define SVM_KER_POLY 2

#include "example.h"
#include "v_array.h"
#include "vw.h"

using namespace std;

namespace KSVM
{

  struct svm_params;

  struct svm_example : public VW::flat_example {
    v_array<float> krow;

    ~svm_example();
    svm_example(VW::flat_example *fec); 
    int compute_kernels(svm_params *params);
    int clear_kernels();
  };

  struct svm_model{    
    size_t num_support;
    v_array<svm_example*> support_vec;
    v_array<float> alpha;
    v_array<float> delta;
  };

  void free_svm_model(svm_model*); 

  learner setup(vw &all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file);
  void driver(vw* all, void* data);
  void learn(void* d, example* ec);
  void finish(void* d);

}
