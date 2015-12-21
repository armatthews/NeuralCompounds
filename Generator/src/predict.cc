#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <fstream>
#include <csignal>

#include "bitext.h"
#include "encdec.h"
#include "decoder.h"
#include "utils.h"

using namespace cnn;
using namespace std;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

void trim(vector<string>& tokens, bool removeEmpty) {
  for (unsigned i = 0; i < tokens.size(); ++i) {
    boost::algorithm::trim(tokens[i]);
    if (tokens[i].length() == 0 && removeEmpty) {
      tokens.erase(tokens.begin() + i);
      --i;
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: cat source.txt | " << argv[0] << " model" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);
  cnn::Initialize(argc, argv);

  Model* cnn_model;
  EncoderDecoderModel* generator;
  Dict source_vocab;
  Dict target_vocab;

  const string model_filename = argv[1];
  ifstream model_file(model_filename);
  if (!model_file.is_open()) {
    cerr << "ERROR: Unable to open " << model_filename << endl;
    exit(1);
  }
  boost::archive::text_iarchive ia(model_file);

  ia & source_vocab;
  ia & target_vocab;
  source_vocab.Freeze();
  target_vocab.Freeze();

  cnn_model = new Model();
  generator = new EncoderDecoderModel(*cnn_model, source_vocab.size(), target_vocab.size());

  ia & *generator;
  ia & *cnn_model;

  Decoder decoder({generator});

  WordId ksSOS = source_vocab.Convert("<s>");
  WordId ksEOS = source_vocab.Convert("</s>");
  WordId ktSOS = target_vocab.Convert("<s>");
  WordId ktEOS = target_vocab.Convert("</s>");

  unsigned beam_size = 10;
  unsigned max_length = 100;
  unsigned kbest_size = 3; 
  decoder.SetParams(max_length, ktSOS, ktEOS);

  string line;
  while(getline(cin, line)) {
    vector<string> parts = tokenize(line, "|||");
    trim(parts, false);

    vector<string> tokens = tokenize(parts[0], " ");
    trim(tokens, true);

    vector<WordId> source(tokens.size());
    for (unsigned i = 0; i < tokens.size(); ++i) {
      source[i] = source_vocab.Convert(tokens[i]);
    }
    source.insert(source.begin(), ksSOS);
    source.insert(source.end(), ksEOS);

    cerr << "Read source sentence: " << boost::algorithm::join(tokens, " ") << endl;
    if (parts.size() > 1) {
      vector<string> reference = tokenize(parts[1], " ");
      trim(reference, true);
      cerr << "  Read reference: " << boost::algorithm::join(reference, " ") << endl;
    }

    ComputationGraph cg;
    KBestList<vector<WordId> > kbest = decoder.TranslateKBest(source, kbest_size, beam_size, cg);
    for (auto& scored_hyp : kbest.hypothesis_list()) {
      double score = scored_hyp.first;
      vector<WordId> hyp = scored_hyp.second;
      vector<string> words(hyp.size());
      for (unsigned i = 0; i < hyp.size(); ++i) {
        words[i] = target_vocab.Convert(hyp[i]);
      }
      string translation = boost::algorithm::join(words, " ");
      cout << score << "\t" << translation << endl;
    }

    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}
