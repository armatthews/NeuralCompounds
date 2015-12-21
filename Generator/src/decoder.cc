#include <iostream>
#include "decoder.h"
#include "utils.h"

Decoder::Decoder(Generator* model) {
  models.push_back(model);
}

Decoder::Decoder(const vector<Generator*>& models) : models(models) {
  assert (models.size() > 0);
}

void Decoder::SetParams(unsigned max_length, WordId kSOS, WordId kEOS) {
  this->max_length = max_length;
  this->kSOS = kSOS;
  this->kEOS = kEOS;
}

vector<WordId> Decoder::Translate(const vector<WordId>& source, unsigned beam_size, ComputationGraph& cg) {
  KBestList<vector<WordId>> kbest = TranslateKBest(source, 1, beam_size, cg);
  return kbest.hypothesis_list().begin()->second;
}

KBestList<vector<WordId>> Decoder::TranslateKBest(const vector<WordId>& source, unsigned K, unsigned beam_size, ComputationGraph& cg) {
  KBestList<vector<WordId> > completed_hyps(K);
  KBestList<vector<PartialHypothesis>> top_hyps(beam_size);
  const unsigned source_length = source.size();

  // XXX: We're storing the same word sequence N times
  vector<PartialHypothesis> initial_partial_hyps(models.size());
  for (unsigned i = 0; i < models.size(); ++i) {
    models[i]->Encode(source, cg);
    initial_partial_hyps[i] = {{kSOS}, models[i]->output_builder.back(), models[i]->output_builder.state()};
  }
  top_hyps.add(0.0, initial_partial_hyps);

  // Invariant: each element in top_hyps should have a length of "t"
  for (unsigned t = 1; t <= max_length; ++t) {
    KBestList<vector<PartialHypothesis>> new_hyps(beam_size);
    for (auto scored_hyp : top_hyps.hypothesis_list()) {
      double score = scored_hyp.first;
      vector<PartialHypothesis>& hyp = scored_hyp.second;
      assert (hyp[0].words.size() == t);
      WordId prev_word = hyp[0].words.back();

      vector<Expression> model_log_output_distributions(models.size());
      for (unsigned i = 0; i < models.size(); ++i) {
        Generator* model = models[i];
        const MLP& final_mlp = model->GetFinalMLP(cg);
        model_log_output_distributions[i] = model->ComputeNormalizedLogOutputDistribution(hyp[i].output_state, final_mlp, cg);
      }

      Expression overall_distribution = sum(model_log_output_distributions) / models.size();

      if (models.size() > 1) {
        overall_distribution = log(softmax(overall_distribution)); // Renormalize
      }
      vector<float> dist = as_vector(cg.incremental_forward());

      // Take the K best-looking words
      KBestList<WordId> best_words(beam_size);
      for (unsigned j = 0; j < dist.size(); ++j) {
        best_words.add(dist[j], j);
      }

      // For each of those K words, add it to the current hypothesis, and add the
      // resulting hyp to our kbest list, unless the new word is </s>,
      // in which case we add the new hyp to the list of completed hyps.
      for (pair<double, WordId> p : best_words.hypothesis_list()) {
        double word_score = p.first;
        WordId word = p.second;
        double new_score = score + word_score;
        vector<PartialHypothesis> new_model_hyps(models.size());
        for (unsigned i = 0; i < models.size(); ++i) {
          Expression output_state = models[i]->AddOutputWord(word, hyp[i].rnn_pointer, cg);
          PartialHypothesis new_hyp = {hyp[i].words, output_state, models[i]->output_builder.state()};
          new_hyp.words.push_back(word);
          new_model_hyps[i] = new_hyp;
        }

        if (t == max_length || word == kEOS) {
          completed_hyps.add(new_score, new_model_hyps[0].words);
        }
        else {
          new_hyps.add(new_score, new_model_hyps);
        }
      }
    }
    top_hyps = new_hyps;
  }
  return completed_hyps;
}
