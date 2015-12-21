#include "cnn/mp.h"
#include "cnn/dict.h"
#include "input_sentence.h"
#include "classifier.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

// This function lets us elegantly handle the user pressing ctrl-c.
// We set a global flag, which causes the training loops to clean up
// and break. In particular, this allows models to be saved to disk
// before actually exiting the program.
bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    cerr << "Exiting..." << endl;
    exit(1);
  }
  else {
    cerr << "Ctrl-c pressed!" << endl;
    ctrlc_pressed = true;
    cnn::mp::stop_requested = true;
  }
}

void Serialize(Dict& dict, Dict& pos_dict, CompoundClassifier& compound_model, Model& cnn_model) {
  int r = ftruncate(fileno(stdout), 0);
  if (r != 0) {
    //cerr << "WARNING: Unable to truncate stdout. Error " << errno << endl;
  }
  fseek(stdout, 0, SEEK_SET);

  boost::archive::text_oarchive oa(cout);
  oa & dict;
  oa & pos_dict;
  oa & compound_model;
  oa & cnn_model;
}

vector<string> tokenize(string input, string delimiter, unsigned max_times) {
  vector<string> tokens;
  //tokens.reserve(max_times);
  size_t last = 0;
  size_t next = 0;
  while ((next = input.find(delimiter, last)) != string::npos && tokens.size() < max_times) {
    tokens.push_back(input.substr(last, next-last));
    last = next + delimiter.length();
  }
  tokens.push_back(input.substr(last));
  return tokens;
}

vector<string> tokenize(string input, string delimiter) {
  return tokenize(input, delimiter, input.length());
}

vector<string> tokenize(string input, char delimiter) {
  return tokenize(input, string(1, delimiter));
}


bool ReadNextCompound(ifstream& f, unsigned* next_compound_line, Span* next_compound_span) {
  string line;
  if (!getline(f, line)) {
    return false;
  }
  vector<string> parts = tokenize(line, "\t");
  *next_compound_line = (unsigned)atoi(parts[0].c_str());
  vector<string> indices = tokenize(parts[2], " ");
  unsigned m = -1;
  unsigned M = 0;
  for (string index : indices) {
    unsigned i = (unsigned)atoi(index.c_str());
    m = (i < m) ? i : m;
    M = (i > M) ? i : M;
  }
  *next_compound_span = make_tuple(m, M + 1);
  return true;
}

vector<InputSentence>* ReadData(const string& sentence_filename, const string& pos_filename, const string& compound_filename, Dict* vocab, Dict* pos_vocab) {
  assert (vocab != NULL);
  assert (pos_vocab != NULL);
  vector<InputSentence>* data = new vector<InputSentence>();
  ifstream sentence_file(sentence_filename);
  ifstream pos_file(pos_filename);
  ifstream compound_file(compound_filename);
  if (!sentence_file.is_open() || !pos_file.is_open() || !compound_file.is_open()) {
    return nullptr;
  }

  string sentence_line;
  string pos_line;
  string compound_line;
  unsigned next_compound_line;
  Span next_compound_span;
  bool more_compounds = ReadNextCompound(compound_file, &next_compound_line, &next_compound_span);

  unsigned i = 0;
  while (getline(sentence_file, sentence_line)) {
    cerr << i << "\r";
    ++i;
    data->push_back(InputSentence());
    InputSentence& input_sentence = data->back();
    if (!getline(pos_file, pos_line)) {
     cerr << "POS tag file (" << pos_filename << ") contains fewer lines than sentences file (" << sentence_filename << ")!" << endl;
     return nullptr;
    }
    //cout << sentence_line << endl << pos_line << endl;
    vector<string> sentence = tokenize(sentence_line, " ");
    vector<string> pos_tags = tokenize(pos_line, " ");
    if (sentence.size() != pos_tags.size()) {
      cerr << "Mismatch in number of words and POS tags on line " << i << endl;
      return nullptr;
    }

    for (string word : sentence) {
      input_sentence.sentence.push_back(vocab->Convert(word));
    }

    for (string pos_tag : pos_tags) {
      input_sentence.pos_tags.push_back(pos_vocab->Convert(pos_tag));
    }

    while (more_compounds && next_compound_line == i) {
      input_sentence.compound_spans.push_back(next_compound_span);
      //cout << "Compound from " << get<0>(next_compound_span) << " to " << get<1>(next_compound_span) << endl;
      more_compounds = ReadNextCompound(compound_file, &next_compound_line, &next_compound_span);
    }
    //cout << endl;
  }

  if (getline(pos_file, pos_line)) {
    cerr << "POS tag file (" << pos_filename << ") contains more lines than sentences file (" << sentence_filename << ")!" << endl;
    return nullptr;
  }
  if (more_compounds) {
    cerr << "Compounds exist in the compounds file (" << compound_filename << ") with indices longer than the length of the sentences file!" << endl;
    return nullptr;
  }
  cerr << "Read " << data->size() << " sentences from " << sentence_filename << endl;
  return data;
}

Trainer* CreateTrainer(Model& model, const po::variables_map& vm) {
  double regularization_strength = vm["regularization"].as<double>();
  double eta_decay = vm["eta_decay"].as<double>();
  bool clipping_enabled = (vm.count("no_clipping") == 0);
  unsigned learner_count = vm.count("sgd") + vm.count("momentum") + vm.count("adagrad") + vm.count("adadelta") + vm.count("rmsprop") + vm.count("adam");
  if (learner_count > 1) {
    cerr << "Invalid parameters: Please specify only one learner type.";
    exit(1);
  }

  Trainer* trainer = NULL;
  if (vm.count("momentum")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.01;
    double momentum = vm["momentum"].as<double>();
    trainer = new MomentumSGDTrainer(&model, regularization_strength, learning_rate, momentum);
  }
  else if (vm.count("adagrad")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    trainer = new AdagradTrainer(&model, regularization_strength, learning_rate, eps);
  }
  else if (vm.count("adadelta")) {
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-6;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new AdadeltaTrainer(&model, regularization_strength, eps, rho);
  }
  else if (vm.count("rmsprop")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new RmsPropTrainer(&model, regularization_strength, learning_rate, eps, rho);
  }
  else if (vm.count("adam")) {
    double alpha = (vm.count("alpha")) ? vm["alpha"].as<double>() : 0.001;
    double beta1 = (vm.count("beta1")) ? vm["beta1"].as<double>() : 0.9;
    double beta2 = (vm.count("beta2")) ? vm["beta2"].as<double>() : 0.999;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-8;
    trainer = new AdamTrainer(&model, regularization_strength, alpha, beta1, beta2, eps);
  }
  else { /* sgd */
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    trainer = new SimpleSGDTrainer(&model, regularization_strength, learning_rate);
  }
  assert (trainer != NULL);

  trainer->eta_decay = eta_decay;
  trainer->clipping_enabled = clipping_enabled;
  return trainer;
}

