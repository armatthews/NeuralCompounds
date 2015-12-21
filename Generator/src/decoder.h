#pragma once
#include "encdec.h"

struct PartialHypothesis {
  vector<WordId> words;
  Expression output_state;
  RNNPointer rnn_pointer;
};

typedef EncoderDecoderModel Generator;

class Decoder {
public:
  explicit Decoder(Generator* model);
  explicit Decoder(const vector<Generator*>& models);
  void SetParams(unsigned max_length, WordId kSOS, WordId kEOS);

  vector<WordId> SampleTranslation(const vector<WordId>& source);
  vector<WordId> Translate(const vector<WordId>& source, unsigned beam_size, ComputationGraph& cg);
  KBestList<vector<WordId>> TranslateKBest(const vector<WordId>& source, unsigned K, unsigned beam_size, ComputationGraph& cg);

private:
  vector<Generator*> models;
  unsigned max_length;
  WordId kSOS;
  WordId kEOS;
};
