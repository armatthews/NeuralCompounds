#include <fstream>
#include "bitext.h"
#include "utils.h"

using namespace std;

void ReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, float* weight) {
  vector<string> parts = tokenize(line, "|||");
  for (unsigned i = 0; i < parts.size(); ++i) {
    parts[i] = strip(parts[i]);
  }
  *weight = 1.0;
  if (parts.size() == 3) {
    *weight = atof(parts[0].c_str());
    parts.erase(parts.begin());
  }
  for (string word : tokenize(parts[0], " ")) {
    s->push_back(sd->Convert(word));
  }
  for (string word : tokenize(parts[1], " ")) {
    t->push_back(td->Convert(word));
  }
}

unsigned Bitext::size() const {
  return sentences.size();
}

bool ReadCorpus(string filename, Bitext& bitext, bool add_bos_eos) {
  ifstream f(filename);
  if (!f.is_open()) {
    return false;
  }

  bitext.source_vocab.Convert("UNK");
  bitext.target_vocab.Convert("UNK");

  WordId sBOS, sEOS, tBOS, tEOS;
  if (add_bos_eos) {
    sBOS = bitext.source_vocab.Convert("<s>");
    sEOS = bitext.source_vocab.Convert("</s>");
    tBOS = bitext.target_vocab.Convert("<s>");
    tEOS = bitext.target_vocab.Convert("</s>");
  }

  for (string line; getline(f, line);) {
    vector<WordId> source;
    vector<WordId> target;
    float weight;
    if (add_bos_eos) {
      source.push_back(sBOS);
      target.push_back(tBOS);
    }
    ReadSentencePair(line, &source, &bitext.source_vocab, &target, &bitext.target_vocab, &weight);
    if (add_bos_eos) {
      source.push_back(sEOS);
      target.push_back(tEOS);
    }
    bitext.sentences.push_back(make_tuple(source, target, weight));
  }
  return true;
}
