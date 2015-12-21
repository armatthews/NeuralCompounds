#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <vector>

#include "train.h"
#include "classifier.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

tuple<Dict*, Dict*, Model*, CompoundClassifier*> LoadModel(string model_filename) {
  ifstream model_file(model_filename);
  if (!model_file.is_open()) {
    cerr << "ERROR: Unable to open " << model_filename << endl;
    exit(1);
  }
  boost::archive::text_iarchive ia(model_file);

  Dict* vocab = new Dict();
  ia & *vocab;
  vocab->Freeze();

  Dict* pos_vocab = new Dict();
  ia & *pos_vocab;
  pos_vocab->Freeze();

  vocab->SetUnk("UNK");
  pos_vocab->SetUnk("NN");

  Model* cnn_model = new Model();
  CompoundClassifier* classifier = new CompoundClassifier();

  ia & *classifier;
  classifier->InitializeParameters(*cnn_model, vocab->size(), pos_vocab->size());

  ia & *cnn_model;

  return make_tuple(vocab, pos_vocab, cnn_model, classifier);
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model file, as output by train")
  ("test_set", po::value<string>()->required(), "Test sentences")
  ("test_pos", po::value<string>()->required(), "Test pos tags")
  ("test_compounds", po::value<string>()->required(), "Test compounds")
  ("down_sample_rate,d", po::value<unsigned>()->default_value(1), "Take only every Nth negative training example")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);
  positional_options.add("test_set", 1);
  positional_options.add("test_pos", 1);
  positional_options.add("test_compounds", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  string model_filename = vm["model"].as<string>();
  const string test_sent_filename = vm["test_set"].as<string>();
  const string test_pos_filename = vm["test_pos"].as<string>();
  const string test_comp_filename = vm["test_compounds"].as<string>();
  cnn::Initialize(argc, argv);

  Dict* vocab = nullptr;
  Dict* pos_vocab = nullptr;
  Model* cnn_model = nullptr;
  CompoundClassifier* classifier = nullptr;
  tie(vocab, pos_vocab, cnn_model, classifier) = LoadModel(model_filename);
  classifier->down_sample_rate = vm["down_sample_rate"].as<unsigned>(); // 75 for FI, 315 for DE

  vector<InputSentence>* test_set = ReadData(test_sent_filename, test_pos_filename, test_comp_filename, vocab, pos_vocab);
  vocab->Freeze();

  string line;
  for (unsigned i = 0; i < test_set->size(); ++i) {
    ComputationGraph cg;
    vector<tuple<Span, double, int>> output = classifier->Predict(test_set->at(i), cg);
    for (unsigned j = 0; j < output.size(); ++j) {
      Span span;
      double prob;
      int label;
      tie(span, prob, label) = output[j];
      cout << i << " ||| " << get<0>(span) << "-" << get<1>(span) << " ||| " << prob << " ||| " << label << endl;
    }

    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}
