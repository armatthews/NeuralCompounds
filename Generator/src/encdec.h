#pragma once
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "bitext.h"
#include "kbestlist.h"
#include "mlp.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class EncoderDecoderModel {
  friend class Decoder;
public:
  EncoderDecoderModel();
  EncoderDecoderModel(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size, bool train = false, bool feed = false);
  void InitializeParameters(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size, bool train);
  void Encode(const vector<WordId>& source, ComputationGraph& cg);
  Expression BuildGraph(const vector<WordId>& source, const vector<WordId>& target, ComputationGraph& cg);
  void NewGraph(ComputationGraph& cg);

protected:
  Expression BuildForwardEncoding(const vector<WordId>& sentence, ComputationGraph& cg);
  Expression BuildReverseEncoding(const vector<WordId>& sentence, ComputationGraph& cg);
  Expression BuildEncoding(const Expression& fwd_encoding, const Expression& rev_encoding) const;

  Expression AddOutputWord(WordId word, ComputationGraph& cg);
  Expression AddOutputWord(WordId word, RNNPointer location, ComputationGraph& cg);
  Expression ComputeOutputDistribution(const MLP& final, ComputationGraph& cg) const;
  Expression ComputeNormalizedLogOutputDistribution(const MLP& final, ComputationGraph& cg) const;
  Expression ComputeOutputDistribution(Expression output_state, const MLP& final, ComputationGraph& cg) const;
  Expression ComputeNormalizedLogOutputDistribution(Expression output_state, const MLP& final, ComputationGraph& cg) const;
  MLP GetFinalMLP(ComputationGraph& cg) const; 

private:
  LSTMBuilder forward_builder, reverse_builder, output_builder;
  vector<Parameters*> forward_initp, reverse_initp;
  vector<Expression> forward_init, reverse_init;
  Parameters* p_mW;
  Parameters* p_mb;
  Expression mW;
  Expression mb;
  LookupParameters* p_Es; // source language word embedding matrix
  LookupParameters* p_Et; // target language word embedding matrix
  Parameters* p_fIH; // "Final" NN (from the tuple (y_{i-1}, s_i, c_i) to the distribution over output words y_i), input->hidden weights
  Parameters* p_fHb; // Same, hidden bias
  Parameters* p_fHO; // Same, hidden->output weights
  Parameters* p_fOb; // Same, output bias

  bool feed = false;
  unsigned lstm_layer_count = 2;
  unsigned embedding_dim = 32; // Dimensionality of both source and target word embeddings. For now these are the same.
  unsigned half_encoding_dim = 128; // Dimensionality of h_forward and h_backward. The full h has twice this dimension.
  unsigned output_hidden_dim = 256; // Dimensionality of s_j, the state just before outputing target word y_j
  unsigned final_hidden_dim = 64; // Dimensionality of the hidden layer in the "final" FFNN

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & lstm_layer_count;
    ar & embedding_dim;
    ar & half_encoding_dim;
    ar & output_hidden_dim;
    ar & final_hidden_dim;
    ar & feed;
  }
};
