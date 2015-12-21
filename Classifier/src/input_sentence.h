#pragma once
#include <vector>
#include <tuple>
using namespace std;

typedef unsigned WordId;
typedef tuple<unsigned, unsigned> Span;

class InputSentence {
public:
  unsigned NumSpans() const;

  vector<WordId> sentence;
  vector<WordId> pos_tags;
  vector<Span> compound_spans;
};
