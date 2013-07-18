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

#include "example.h"
#include "v_array.h"
#include "vw.h"

using namespace std;

namespace KSVM
{

  struct svm_model{    
    int num_support;
    v_array<example*> model;
    v_array<float> alpha;
    v_array<float> delta;
    float maxdelta;    
  }

  

}
