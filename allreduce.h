/*
Copyright (c) 2011 Yahoo! Inc.  All rights reserved.  The copyrights
embodied in the content of this file are licensed under the BSD
(revised) open source license

This implements the allreduce function of MPI.  

 */

#ifndef ALLREDUCE_H
#define ALLREDUCE_H
#include <string>

struct node_socks {
  int parent;
  int children[2];
  ~node_socks();
};

  
using namespace std;

const int buf_size = 1<<18;

void all_reduce(char* buffer, int n, string master_location, int &node_id);

#endif
