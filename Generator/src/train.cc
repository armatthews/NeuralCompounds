#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/mp.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <algorithm>

#include "bitext.h"
#include "encdec.h"
#include "train.h"

using namespace cnn;
using namespace cnn::mp;
using namespace std;
namespace po = boost::program_options;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    cerr << "Exiting..." << endl;
    exit(1);
  }
  else {
    cerr << "Ctrl-c pressed!" << endl;
    ctrlc_pressed = true;
  }
}

class SufficientStats {
public:
  cnn::real loss;
  unsigned word_count;
  unsigned sentence_count;

  SufficientStats() : loss(), word_count(), sentence_count() {}

  SufficientStats(cnn::real loss, unsigned word_count, unsigned sentence_count) : loss(loss), word_count(word_count), sentence_count(sentence_count) {}

  SufficientStats& operator+=(const SufficientStats& rhs) {
    loss += rhs.loss;
    word_count += rhs.word_count;
    sentence_count += rhs.sentence_count;
    return *this;
  }

  friend SufficientStats operator+(SufficientStats lhs, const SufficientStats& rhs) {
    lhs += rhs;
    return lhs;
  }

  bool operator<(const SufficientStats& rhs) {
    return loss < rhs.loss;
  }

  friend std::ostream& operator<< (std::ostream& stream, const SufficientStats& stats) {
    //return stream << "SufficientStats(" << stats.loss << ", " << stats.word_count << ", " << stats.sentence_count << ") = " << exp(stats.loss / stats.word_count);
    return stream << exp(stats.loss / stats.word_count);
  }
};

void Serialize(Bitext& bitext, EncoderDecoderModel& generator, Model& model) {
  int r = ftruncate(fileno(stdout), 0);
  fseek(stdout, 0, SEEK_SET); 

  boost::archive::text_oarchive oa(cout);
  oa & bitext.source_vocab;
  oa & bitext.target_vocab;
  oa << generator;
  oa << model;
}

template<class D>
class Learner : public ILearner<D, SufficientStats> {
public:
  explicit Learner(Bitext* bitext, EncoderDecoderModel& generator, Model& model) : bitext(bitext), generator(generator), model(model) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const D& datum, bool learn) {
    ComputationGraph cg;
    vector<WordId> source = get<0>(datum);
    vector<WordId> target = get<1>(datum);
    float weight = get<2>(datum);
    Expression loss_expr = generator.BuildGraph(source, target, cg) * weight;
    SufficientStats loss(as_scalar(cg.forward()), target.size() - 1, 1);
    if (learn) {
      cg.backward();
    }
    return loss;
  }

  void SaveModel() {
    cerr << "Saving model..." << endl;
    Serialize(*bitext, generator, model);
    cerr << "Done saving model." << endl;
  }
private:
  Bitext* bitext;
  EncoderDecoderModel& generator;
  Model& model;
};

template <class RNG>
void shuffle(Bitext& bitext, RNG& g) {
  vector<unsigned> indices(bitext.size(), 0);
  for (unsigned i = 0; i < bitext.size(); ++i) {
    indices[i] = i;
  }
  shuffle(indices.begin(), indices.end(), g);
  vector<vector<WordId> > source(bitext.size());
  vector<vector<WordId> > target(bitext.size());
  vector<float> weights(bitext.size());
  for (unsigned i = 0; i < bitext.size(); ++i) {
    unsigned j = indices[i];
    source[j] = get<0>(bitext.sentences[i]);
    target[j] = get<1>(bitext.sentences[i]);
    weights[j] = get<2>(bitext.sentences[i]);
  }

  for (unsigned i = 0; i < bitext.size(); ++i) {
    bitext.sentences[i] = make_tuple(source[i], target[i], weights[i]);
  }
}

pair<cnn::real, unsigned> ComputeLoss(Bitext& bitext, EncoderDecoderModel& generator) {
  cnn::real loss = 0.0;
  unsigned word_count = 0;
  for (unsigned i = 0; i < bitext.size(); ++i) {
    vector<WordId> source_sentence = get<0>(bitext.sentences[i]);
    vector<WordId> target_sentence = get<1>(bitext.sentences[i]);
    float weight = get<2>(bitext.sentences[i]);
    word_count += target_sentence.size() - 1; // Minus one for <s>
    ComputationGraph cg;
    generator.BuildGraph(source_sentence, target_sentence, cg);
    double l = as_scalar(cg.forward())* weight;
    loss += l;
    if (ctrlc_pressed) {
      break;
    }
  }
  return make_pair(loss, word_count);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " corpus.txt" << endl;
    cerr << endl;
    exit(1);
  }
  signal (SIGINT, ctrlc_handler);

  po::options_description desc("description");
  desc.add_options()
  ("train_bitext", po::value<string>()->required(), "Training bitext in source ||| target format")
  ("dev_bitext", po::value<string>()->required(), "Dev bitext, used for early stopping")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("batch_size,b", po::value<unsigned>()->default_value(1), "Size of minibatches")
  ("random_seed,r", po::value<unsigned>()->default_value(0), "Random seed. If this value is 0 a seed will be chosen randomly.")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")
  ("feed", "Feed output hidden state back into LSTM at every time step")
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
  ("model", po::value<string>(), "Reload this model and continue learning")
  // End optimizer configuration
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("train_bitext", 1);
  positional_options.add("dev_bitext", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string train_bitext_filename = vm["train_bitext"].as<string>();
  const string dev_bitext_filename = vm["dev_bitext"].as<string>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const unsigned random_seed = vm["random_seed"].as<unsigned>();
  const unsigned batch_size = vm["batch_size"].as<unsigned>();
  const unsigned num_children = vm["cores"].as<unsigned>();
  const unsigned feed = vm.count("feed") > 0;

  cnn::Initialize(argc, argv, random_seed, true);
  std::mt19937 rndeng(42);
  Model* cnn_model = new Model();
  EncoderDecoderModel* generator = new EncoderDecoderModel();

  Bitext train_bitext;
  if (vm.count("model")) {
    const string model_filename = vm["model"].as<string>();
    ifstream model_file(model_filename);
    if (!model_file.is_open()) {
      cerr << "ERROR: Unable to open " << model_filename << endl;
      exit(1);
    }
    boost::archive::text_iarchive ia(model_file);

    ia & train_bitext.source_vocab;
    ia & train_bitext.target_vocab;
    train_bitext.source_vocab.Freeze();
    train_bitext.target_vocab.Freeze();

    ia & *generator;
    generator->InitializeParameters(*cnn_model, train_bitext.source_vocab.size(), train_bitext.target_vocab.size(), true);
    ia & *cnn_model;
  }

  ReadCorpus(train_bitext_filename, train_bitext, true);
  cerr << "Read " << train_bitext.size() << " lines from " << train_bitext_filename << endl;
  cerr << "Vocab size: " << train_bitext.source_vocab.size() << "/" << train_bitext.target_vocab.size() << endl; 
  if (!vm.count("model")) {
    generator = new EncoderDecoderModel(*cnn_model, train_bitext.source_vocab.size(), train_bitext.target_vocab.size(), true, feed);
  }

  Trainer* sgd = CreateTrainer(*cnn_model, vm);
  Bitext dev_bitext; 
  // TODO: The vocabulary objects really need to be tied. This is a really ghetto way of doing it
  dev_bitext.source_vocab = train_bitext.source_vocab;
  dev_bitext.target_vocab = train_bitext.target_vocab;
  unsigned initial_source_vocab_size = dev_bitext.source_vocab.size();
  unsigned initial_target_vocab_size = dev_bitext.target_vocab.size();

  dev_bitext.source_vocab.Freeze();
  dev_bitext.source_vocab.SetUnk("UNK");

  dev_bitext.target_vocab.Freeze();
  dev_bitext.target_vocab.SetUnk("UNK");

  ReadCorpus(dev_bitext_filename, dev_bitext, true);
  // Make sure the vocabulary sizes didn't change. If the dev set contains any words not in the training set, that's a problem!
  assert (initial_source_vocab_size == dev_bitext.source_vocab.size());
  assert (initial_target_vocab_size == dev_bitext.target_vocab.size());
  cerr << "Read " << dev_bitext.size() << " lines from " << dev_bitext_filename << endl;
  cerr << "Vocab size: " << dev_bitext.source_vocab.size() << "/" << dev_bitext.target_vocab.size() << endl;

  unsigned dev_frequency = 10000;
  unsigned report_frequency = 50;
  Learner<Bitext::SentencePair> learner(&train_bitext, *generator, *cnn_model);
  RunMultiProcess<Bitext::SentencePair>(num_children, &learner, sgd, train_bitext.sentences, dev_bitext.sentences, num_iterations, dev_frequency, report_frequency);

  /*cerr << "Training model...\n";
  unsigned minibatch_count = 0;
  const unsigned minibatch_size = std::min(batch_size, train_bitext.size());
  const unsigned report_frequency = 50;
  cnn::real best_dev_loss = numeric_limits<cnn::real>::max();
  for (unsigned iteration = 0; iteration < num_iterations; iteration++) {
    unsigned word_count = 0;
    unsigned tword_count = 0;
    shuffle(train_bitext, rndeng);
    double loss = 0.0;
    double tloss = 0.0;
    for (unsigned i = 0; i < train_bitext.size(); ++i) {
      //cerr << "Reading sentence pair #" << i << endl;
      vector<WordId>& source_sentence = get<0>(train_bitext.sentences[i]);
      vector<WordId>& target_sentence = get<1>(train_bitext.sentences[i]);
      float weight = get<2>(train_bitext.sentences[i]);
      word_count += target_sentence.size() - 1; // Minus one for <s>
      tword_count += target_sentence.size() - 1; // Minus one for <s>
      ComputationGraph cg;
      Expression loss_expr = generator->BuildGraph(source_sentence, target_sentence, cg) * weight;
      double l = as_scalar(cg.forward());
      loss += l;
      tloss += l;
      cg.backward();
      if (i % report_frequency == report_frequency - 1) {
        float fractional_iteration = (float)iteration + ((float)i / train_bitext.size());
        cerr << "--" << fractional_iteration << " loss: " << tloss << " (perp=" << exp(tloss/tword_count) << ")" << endl;
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
    if (ctrlc_pressed) {
      break;
    }
    auto dev_loss = ComputeLoss(dev_bitext, *generator);
    cerr << "Iteration " << iteration + 1 << " loss: " << loss << " (perp=" << exp(loss/word_count) << ")" << " dev loss: " << dev_loss.first << " (perp: " << exp(dev_loss.first / dev_loss.second) << ")" << endl;
    sgd->update_epoch();
    if (dev_loss.first <= best_dev_loss) {
      cerr << "New best!" << endl;
      Serialize(train_bitext, *generator, *cnn_model);
      best_dev_loss = dev_loss.first;
    }
  }*/

  return 0;
}
