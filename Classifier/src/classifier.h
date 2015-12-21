#pragma once
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "input_sentence.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

struct MLP {
  vector<Expression> i_IH;
  Expression i_Hb;
  Expression i_HO;
  Expression i_Ob;

  Expression Feed(vector<Expression> input) const;
};

class CompoundClassifier {
public:
  CompoundClassifier();
  CompoundClassifier(Model& model, unsigned vocab_size, unsigned num_pos_tags);
  void InitializeParameters(Model& model, unsigned vocab_size, unsigned num_pos_tags);

  vector<tuple<Span, Expression, int>> BuildExpressions(const InputSentence& input_sentence, ComputationGraph& cg);
  Expression BuildGraph(const InputSentence& input_sentence, ComputationGraph& cg);
  vector<tuple<Span, double, int>> Predict(const InputSentence& input_sentence, ComputationGraph& cg);
  MLP GetFinalMLP(ComputationGraph& cg);

  unsigned down_sample_rate = 75;

private:
  LSTMBuilder forward_builder;
  LSTMBuilder reverse_builder;
  LookupParameters* p_word_lookup;
  LookupParameters* p_pos_lookup;

  Parameters* p_fIH;
  Parameters* p_fHb;
  Parameters* p_fHO;
  Parameters* p_fOb;

  unsigned lstm_layer_count = 2;
  unsigned word_embedding_dim = 10;
  unsigned pos_embedding_dim = 10;
  unsigned lstm_hidden_dim = 10;
  unsigned final_hidden_dim = 10;

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & lstm_layer_count;
    ar & word_embedding_dim;
    ar & pos_embedding_dim;
    ar & final_hidden_dim;
  }
};
