#include "mlp.h"

Expression MLP::Feed(vector<Expression> inputs) const {
  assert (inputs.size() == i_IH.size());
  vector<Expression> xs(2 * inputs.size() + 1);
  xs[0] = i_Hb;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    xs[2 * i + 1] = i_IH[i];
    xs[2 * i + 2] = inputs[i];
  }
  Expression hidden1 = affine_transform(xs);
  Expression hidden2 = tanh({hidden1});
  Expression output = affine_transform({i_Ob, i_HO, hidden2});
  return output;
}

