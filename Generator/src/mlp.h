#pragma once
#include <vector>
#include "cnn/cnn.h"
#include "cnn/expr.h"
using namespace std;
using namespace cnn;
using namespace cnn::expr;

// A simple 2 layer MLP
struct MLP {
  vector<Expression> i_IH;
  Expression i_Hb;
  Expression i_HO;
  Expression i_Ob;

  Expression Feed(vector<Expression> input) const;
};
