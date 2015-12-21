#include "input_sentence.h"

unsigned InputSentence::NumSpans() const {
  unsigned span_count = 0;
  for (unsigned i = 2; i <= 5; ++i) {
    span_count += sentence.size() - i + 1;
  }
  return span_count;
}
