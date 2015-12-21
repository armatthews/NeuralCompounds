#include "classifier.h"

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

CompoundClassifier::CompoundClassifier() {
}

CompoundClassifier::CompoundClassifier(Model& model, unsigned vocab_size, unsigned num_pos_tags) {
  InitializeParameters(model, vocab_size, num_pos_tags);
}

void CompoundClassifier::InitializeParameters(Model& model, unsigned vocab_size, unsigned num_pos_tags) {
  forward_builder = LSTMBuilder(lstm_layer_count, word_embedding_dim + pos_embedding_dim, lstm_hidden_dim, &model);
  reverse_builder = LSTMBuilder(lstm_layer_count, word_embedding_dim + pos_embedding_dim, lstm_hidden_dim, &model);

  p_word_lookup = model.add_lookup_parameters(vocab_size, {word_embedding_dim});
  p_pos_lookup = model.add_lookup_parameters(num_pos_tags, {pos_embedding_dim});

  p_fIH = model.add_parameters({final_hidden_dim, 2*lstm_hidden_dim});
  p_fHb = model.add_parameters({final_hidden_dim});
  p_fHO = model.add_parameters({2, final_hidden_dim});
  p_fOb = model.add_parameters({2});
}

MLP CompoundClassifier::GetFinalMLP(ComputationGraph& cg) {
  Expression ih = parameter(cg, p_fIH);
  Expression hb = parameter(cg, p_fHb);
  Expression ho = parameter(cg, p_fHO);
  Expression ob = parameter(cg, p_fOb);
  return {{ih}, hb, ho, ob};
}

vector<tuple<Span, Expression, int>> CompoundClassifier::BuildExpressions(const InputSentence& input_sentence, ComputationGraph& cg) {
  vector<tuple<Span, Expression, int>> output_expressions;
  if (input_sentence.sentence.size() < 2) {
    return output_expressions;
  }

  forward_builder.new_graph(cg);
  reverse_builder.new_graph(cg);
  MLP final_mlp = GetFinalMLP(cg);

  vector<Expression> losses;
  for (int length = 2; length <= 4; length++) {
    for (int start = 0; start <= (int)input_sentence.sentence.size() - length; ++start) {
      int end = start + length;

      int label = 0;
      for (Span span : input_sentence.compound_spans) {
        if (get<0>(span) == (unsigned)start && get<1>(span) == (unsigned)end) {
          label = 1;
        }
      }
      if (label == 0 && rand() % down_sample_rate > 0) {
        continue;
      }

      forward_builder.start_new_sequence();
      reverse_builder.start_new_sequence();
      Expression fwd_embedding;
      Expression rev_embedding;
      vector<Expression> word_representations;
      for (int i = start; i < end; ++i) {
        Expression word_vector = lookup(cg, p_word_lookup, input_sentence.sentence[i]);
        Expression pos_vector = lookup(cg, p_pos_lookup, input_sentence.pos_tags[i]);
        word_representations.push_back(concatenate({word_vector, pos_vector}));
      }
      for (auto it = word_representations.begin(); it != word_representations.end(); ++it) {
        fwd_embedding = forward_builder.add_input(*it);
      }
      for (auto it = word_representations.rbegin(); it != word_representations.rend(); ++it) {      
        rev_embedding = reverse_builder.add_input(*it);
      }
      Expression embedding = concatenate({fwd_embedding, rev_embedding});
      Expression output = final_mlp.Feed({embedding});
      output_expressions.push_back(make_tuple(make_tuple(start, end), output, label));
    }
  }
  return output_expressions;
}

vector<tuple<Span, double, int>> CompoundClassifier::Predict(const InputSentence& input_sentence, ComputationGraph& cg) {
  vector<tuple<Span, double, int>> output;
  vector<tuple<Span, Expression, int>> expressions = BuildExpressions(input_sentence, cg);
  for (unsigned i = 0; i < expressions.size(); ++i) {
    Span span;
    Expression exp;
    int label;
    tie(span, exp, label) = expressions[i];
    Expression probs_exp = softmax(exp);
    cg.incremental_forward();
    vector<float> probs = as_vector(probs_exp.value());
    output.push_back(make_tuple(span, probs[1], label));
  }
  return output;
}

Expression CompoundClassifier::BuildGraph(const InputSentence& input_sentence, ComputationGraph& cg) {
  vector<tuple<Span, Expression, int>> expressions = BuildExpressions(input_sentence, cg);
  vector<Expression> losses;
  for (unsigned i = 0; i < expressions.size(); ++i) {  
    Span span;
    Expression exp;
    int label;
    tie(span, exp, label) = expressions[i];
    losses.push_back(pickneglogsoftmax(exp, label));
  }
  if (losses.size() == 0) {
    return input(cg, 0.0);
  }
  return sum(losses);
}
