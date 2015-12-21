#include <queue>
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"

#include "bitext.h"
#include "encdec.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

EncoderDecoderModel::EncoderDecoderModel() : feed(false) {}

EncoderDecoderModel::EncoderDecoderModel(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size, bool train, bool feed) : feed(feed) {
  InitializeParameters(model, src_vocab_size, tgt_vocab_size, train);
}

void EncoderDecoderModel::InitializeParameters(Model& model, unsigned src_vocab_size, unsigned tgt_vocab_size, bool train) {
  forward_builder = LSTMBuilder(lstm_layer_count, embedding_dim, half_encoding_dim, &model);
  reverse_builder = LSTMBuilder(lstm_layer_count, embedding_dim, half_encoding_dim, &model);
  output_builder = LSTMBuilder(lstm_layer_count, embedding_dim + (feed ? output_hidden_dim : 0), output_hidden_dim, &model);
  p_Es = model.add_lookup_parameters(src_vocab_size, {embedding_dim});
  p_Et = model.add_lookup_parameters(tgt_vocab_size, {embedding_dim});

  p_fIH = model.add_parameters({final_hidden_dim, output_hidden_dim});
  p_fHb = model.add_parameters({final_hidden_dim});
  p_fHO = model.add_parameters({tgt_vocab_size, final_hidden_dim});
  p_fOb = model.add_parameters({tgt_vocab_size});

  p_mW = model.add_parameters({output_builder.num_h0_components() * output_hidden_dim, 2 * half_encoding_dim});
  p_mb = model.add_parameters({output_builder.num_h0_components() * output_hidden_dim});

  for (unsigned i = 0; i < lstm_layer_count; ++i) {
    forward_initp.push_back(model.add_parameters({half_encoding_dim}));
    forward_initp.push_back(model.add_parameters({half_encoding_dim}));
    reverse_initp.push_back(model.add_parameters({half_encoding_dim}));
    reverse_initp.push_back(model.add_parameters({half_encoding_dim}));
  }

  if (train) {
    forward_builder.set_dropout(0.2);
    reverse_builder.set_dropout(0.2);
    output_builder.set_dropout(0.2);
  }
  
}

Expression EncoderDecoderModel::BuildForwardEncoding(const vector<WordId>& sentence, ComputationGraph& cg) {
  assert (sentence.size() > 0);
  forward_builder.new_graph(cg);
  forward_builder.start_new_sequence(forward_init);
  vector<Expression> forward_annotations(sentence.size());
  Expression r;
  for (unsigned t = 0; t < sentence.size(); ++t) {
    Expression i_x_t = lookup(cg, p_Es, sentence[t]);
    r = forward_builder.add_input(i_x_t);
  }
  return r;
}

Expression EncoderDecoderModel::BuildReverseEncoding(const vector<WordId>& sentence, ComputationGraph& cg) {
  assert (sentence.size() > 0);
  reverse_builder.new_graph(cg);
  reverse_builder.start_new_sequence(reverse_init);
  Expression r;
  for (unsigned t = sentence.size(); t > 0; ) {
    t--;
    Expression i_x_t = lookup(cg, p_Es, sentence[t]);
    r = reverse_builder.add_input(i_x_t);
  }
  return r;
}

Expression EncoderDecoderModel::BuildEncoding(const Expression& fwd_embedding, const Expression& rev_embedding) const {
  return concatenate({fwd_embedding, rev_embedding});
}

Expression EncoderDecoderModel::AddOutputWord(WordId word, ComputationGraph& cg) {
  return AddOutputWord(word, output_builder.state(), cg);
}

Expression EncoderDecoderModel::AddOutputWord(WordId word, RNNPointer location, ComputationGraph& cg) {
  Expression word_embedding = lookup(cg, p_Et, word);
  Expression input = feed ? concatenate({word_embedding, output_builder.h0.back()}) : word_embedding;
  Expression new_output_embedding = output_builder.add_input(location, input);
  return new_output_embedding;
}

Expression EncoderDecoderModel::ComputeOutputDistribution(Expression output_state, const MLP& final, ComputationGraph& cg) const {
  Expression output_dist = final.Feed({output_state}); 
  return output_dist;
}

Expression EncoderDecoderModel::ComputeOutputDistribution(const MLP& final, ComputationGraph& cg) const {
  return ComputeOutputDistribution(output_builder.back(), final, cg);
}

Expression EncoderDecoderModel::ComputeNormalizedLogOutputDistribution(Expression output_state, const MLP& final, ComputationGraph& cg) const {
  Expression output_dist = ComputeOutputDistribution(output_state, final, cg);
  return log(softmax(output_dist));
}

Expression EncoderDecoderModel::ComputeNormalizedLogOutputDistribution(const MLP& final, ComputationGraph& cg) const {
  return ComputeNormalizedLogOutputDistribution(output_builder.back(), final, cg);
}

MLP EncoderDecoderModel::GetFinalMLP(ComputationGraph& cg) const {
  Expression i_fIH = parameter(cg, p_fIH);
  Expression i_fHb = parameter(cg, p_fHb);
  Expression i_fHO = parameter(cg, p_fHO);
  Expression i_fOb = parameter(cg, p_fOb);
  MLP final_mlp = {{i_fIH}, i_fHb, i_fHO, i_fOb};
  return final_mlp;
}

void EncoderDecoderModel::NewGraph(ComputationGraph& cg) {
  forward_init.clear();
  for (Parameters* p : forward_initp) {
    forward_init.push_back(parameter(cg, p));
  }

  reverse_init.clear();
  for (Parameters* p : reverse_initp) {
    reverse_init.push_back(parameter(cg, p));
  }

  mW = parameter(cg, p_mW);
  mb = parameter(cg, p_mb);
};

void EncoderDecoderModel::Encode(const vector<WordId>& source, ComputationGraph& cg) {
  NewGraph(cg);
  output_builder.new_graph(cg);

  Expression encoding = BuildEncoding(BuildForwardEncoding(source, cg), BuildReverseEncoding(source, cg));

  Expression output_init_all = tanh(affine_transform({mb, mW, encoding}));
  vector<Expression> output_init;
  unsigned start = 0;
  for (unsigned i = 0; i < output_builder.num_h0_components(); ++i) {
    Expression piece = pickrange(output_init_all, start, start + output_hidden_dim);
    start += output_hidden_dim;
    output_init.push_back(piece);
  }
  output_builder.start_new_sequence(output_init);
}

Expression EncoderDecoderModel::BuildGraph(const vector<WordId>& source, const vector<WordId>& target, ComputationGraph& cg) {
  // Target should always contain at least <s> and </s>
  assert (target.size() > 2);
  const unsigned kBOS = 1;
  const unsigned kEOS = 2;
  assert (target[0] == kBOS);
  assert (target.back() == kEOS);

  Encode(source, cg);
  MLP final = GetFinalMLP(cg);

  vector<Expression> losses;
  for (unsigned t = 1; t < target.size(); ++t) {
    Expression dist = ComputeOutputDistribution(final, cg);
    Expression word_loss = pickneglogsoftmax(dist, target[t]);
    losses.push_back(word_loss);
    AddOutputWord(target[t], cg);
  }

  Expression total_loss = sum(losses);
  return total_loss;
}
