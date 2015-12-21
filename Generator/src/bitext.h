#pragma once
#include <vector>
#include "cnn/dict.h"

using namespace std;
using namespace cnn;

typedef int WordId;

struct Bitext {
  typedef vector<WordId> SourceSentence;
  typedef vector<WordId> TargetSentence;
  typedef tuple<SourceSentence, TargetSentence, float> SentencePair;
  vector<SentencePair> sentences;
  Dict source_vocab;
  Dict target_vocab;

  unsigned size() const;
};

bool ReadCorpus(string filename, Bitext& bitext, bool add_bos_eos);
