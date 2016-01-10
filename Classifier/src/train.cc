#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <random>
#include <memory>
#include <algorithm>

#include "classifier.h"
#include "train.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

pair<cnn::real, unsigned> ComputeLoss(const vector<InputSentence>& data, CompoundClassifier& model) {
  cnn::real loss = 0.0;
  unsigned span_count = 0;
  for (unsigned i = 0; i < data.size(); ++i) {
    ComputationGraph cg;
    model.BuildGraph(data[i], cg);
    span_count += data[i].NumSpans();
    double l = as_scalar(cg.forward());
    loss += l;
    if (ctrlc_pressed) {
      break;
    }
  }
  return make_pair(loss, span_count);
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);

  po::options_description desc("description");
  desc.add_options()
  ("training_set", po::value<string>()->required(), "Training sentences")
  ("training_pos", po::value<string>()->required(), "Training pos tags")
  ("training_compounds", po::value<string>()->required(), "Training compounds, as output by findCompounds.py")
  ("dev_set", po::value<string>()->required(), "Dev sentences, used for early stopping")
  ("dev_pos", po::value<string>()->required(), "Dev pos tags")
  ("dev_compounds", po::value<string>()->required(), "Dev compounds")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("batch_size,b", po::value<unsigned>()->default_value(1), "Size of minibatches")
  ("random_seed,r", po::value<unsigned>()->default_value(0), "Random seed. If this value is 0 a seed will be chosen randomly.")
  ("down_sample_rate,d", po::value<unsigned>()->default_value(1), "Take only every Nth negative training example")
  ("max_length,n", po::value<unsigned>()->default_value(4), "Max length source span that can compound")
  // Optimizer configuration
  ("sgd", "Use SGD for optimization")
  ("momentum", po::value<double>(), "Use SGD with this momentum value")
  ("adagrad", "Use Adagrad for optimization")
  ("adadelta", "Use Adadelta for optimization")
  ("rmsprop", "Use RMSProp for optimization")
  ("adam", "Use Adam for optimization")
  ("learning_rate", po::value<double>(), "Learning rate for optimizer (SGD, Adagrad, Adadelta, and RMSProp only)")
  ("alpha", po::value<double>(), "Alpha (Adam only)")
  ("beta1", po::value<double>(), "Beta1 (Adam only)")
  ("beta2", po::value<double>(), "Beta2 (Adam only)")
  ("rho", po::value<double>(), "Moving average decay parameter (RMSProp and Adadelta only)")
  ("epsilon", po::value<double>(), "Epsilon value for optimizer (Adagrad, Adadelta, RMSProp, and Adam only)")
  ("regularization", po::value<double>()->default_value(0.0), "L2 Regularization strength")
  ("eta_decay", po::value<double>()->default_value(0.05), "Learning rate decay rate (SGD only)")
  ("no_clipping", "Disable clipping of gradients")
  // End optimizer configuration
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("training_set", 1);
  positional_options.add("training_pos", 1);
  positional_options.add("training_compounds", 1);
  positional_options.add("dev_set", 1);
  positional_options.add("dev_pos", 1);
  positional_options.add("dev_compounds", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string train_sent_filename = vm["training_set"].as<string>();
  const string train_pos_filename = vm["training_pos"].as<string>();
  const string train_comp_filename = vm["training_compounds"].as<string>();
  const string dev_sent_filename = vm["dev_set"].as<string>();
  const string dev_pos_filename = vm["dev_pos"].as<string>();
  const string dev_comp_filename = vm["dev_compounds"].as<string>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const unsigned random_seed = vm["random_seed"].as<unsigned>();
  const unsigned minibatch_size = vm["batch_size"].as<unsigned>();
  const unsigned max_length = vm["max_length"].as<unsigned>();

  cnn::Initialize(argc, argv, random_seed);
  std::mt19937 rndeng(42);
  CompoundClassifier* classifier_model = new CompoundClassifier(max_length);
  classifier_model->down_sample_rate = vm["down_sample_rate"].as<unsigned>();
  Model* cnn_model = new Model();
  Dict vocab;
  Dict pos_vocab;

  vocab.Convert("UNK");
  vector<InputSentence>* training_set = ReadData(train_sent_filename, train_pos_filename, train_comp_filename, &vocab, &pos_vocab);
  assert (minibatch_size <= training_set->size());
  //vocab.Freeze();
  vector<InputSentence>* dev_set = ReadData(dev_sent_filename, dev_pos_filename, dev_comp_filename, &vocab, &pos_vocab);
  cerr << "Vocab size: " << vocab.size() << endl;
  cerr << "POS vocab size: " << pos_vocab.size() << endl;

  classifier_model->InitializeParameters(*cnn_model, vocab.size(), pos_vocab.size());
  Trainer* sgd = CreateTrainer(*cnn_model, vm);

  cerr << "Training model...\n";
  unsigned minibatch_count = 0;
  const unsigned report_frequency = 500;
  cnn::real best_dev_loss = numeric_limits<cnn::real>::max();
  for (unsigned iteration = 0; iteration < num_iterations; iteration++) {
    unsigned word_count = 0;
    unsigned tword_count = 0;
    random_shuffle(training_set->begin(), training_set->end());
    double loss = 0.0;
    double tloss = 0.0;
    for (unsigned i = 0; i < training_set->size(); ++i) { 
      // These braces cause cg to go out of scope before we ever try to call
      // ComputeLoss() on the dev set. Without them, ComputeLoss() tries to
      // create a second ComputationGraph, which makes CNN quite unhappy.
      {
        ComputationGraph cg;
        InputSentence& example = training_set->at(i);
        classifier_model->BuildGraph(example, cg);
        unsigned sent_word_count = example.NumSpans();
        word_count += sent_word_count;
        tword_count += sent_word_count;
        double sent_loss = as_scalar(cg.forward());
        loss += sent_loss;
        tloss += sent_loss;
        cg.backward();
      }
      if (i % report_frequency == report_frequency - 1) {
        float fractional_iteration = (float)iteration + ((float)(i + 1) / training_set->size());
        cerr << "--" << fractional_iteration << "     perp=" << exp(tloss/tword_count * (classifier_model->down_sample_rate + 1) / 2.0) << endl;
        cerr.flush();
        tloss = 0;
        tword_count = 0;
      }
      if (++minibatch_count == minibatch_size) {
        sgd->update(1.0 / minibatch_size);
        minibatch_count = 0;
      }
      if (ctrlc_pressed) {
        break;
      }
    }
    //sgd->update_epoch();
    cerr << "##" << (float)(iteration + 1) << "     perp=" << exp(loss / word_count * (classifier_model->down_sample_rate + 1) / 2.0) << endl;
    if (!ctrlc_pressed) {
      auto dev_loss = ComputeLoss(*dev_set, *classifier_model);
      cnn::real dev_perp = exp(dev_loss.first / dev_loss.second * (classifier_model->down_sample_rate + 1) / 2.0);
      bool new_best = dev_loss.first <= best_dev_loss;
      cerr << "**" << iteration + 1 << " dev perp: " << dev_perp << (new_best ? " (New best!)" : "") << endl;
      cerr.flush();
      if (new_best) {
        Serialize(vocab, pos_vocab, *classifier_model, *cnn_model);
        best_dev_loss = dev_loss.first;
      }
    }

    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}
