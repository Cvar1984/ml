<?php

require './vendor/autoload.php';
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Tokenizers\NGram;
use Rubix\ML\Transformers\MultibyteTextNormalizer;
use Rubix\ML\Transformers\RegexFilter;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\Persisters\Filesystem;
use League\Csv\Reader;
use League\Csv\Statement;

ini_set('memory_limit', '-1');

//load the CSV document from a stream
$rubixModelPath = 'assets/rubix/sentiment.rbx';
$labeledDataPath = 'assets/dataset/sentiment.csv';
$progresDataPath = 'assets/dataset/progress.csv';

$csv = Reader::createFromPath($labeledDataPath);
$csv->setDelimiter(',');
$csv->setHeaderOffset(0);

//build a statement
$stmt = Statement::create()
    ->offset(0)
    ->limit($argv[1]); // limit rows to be loaded. current rows 96328

$records = $stmt->process($csv);

$samples = $labels = [];

foreach ($records as $record) {
    $samples[] = [$record['review_sangat_singkat']];
    $labels[] = $record['label'];
}

$logger = new Screen();
$logger->info('Loading data into memory');
$dataset = new Labeled($samples, $labels);

$nodeSize = 100; // The number of nodes in hidden layers.
$batchSize = 256; // The number of training samples to process at a time
$vocabularyMaxSize = PHP_INT_MAX; // The maximum number of unique tokens to embed into each document vector.
$documentMinSize = 0.00008; // The minimum proportion of documents a word must appear in to be added to the vocabulary.
$documentMaxSize = 0.4; // The maximum proportion of documents a word can appear in to be added to the vocabulary.
$nGramMinSize = 1; // The minimum number of contiguous words to a token.
$nGramMaxSize = 2; // The maximum number of contiguous words to a token.
$alphaSize = 0.0; // The amount of L2 regularization applied to the weights.
$estimator = new PersistentModel(
    new Pipeline([
        new MultibyteTextNormalizer(), // require ext-mbstring
        new WordCountVectorizer(
            $vocabularyMaxSize,
            $documentMinSize,
            $documentMaxSize,
            new NGram($nGramMinSize, $nGramMaxSize)
        ),
        new TfIdfTransformer(),
        new ZScaleStandardizer(),
        new RegexFilter([
            RegexFilter::EXTRA_CHARACTERS,
            RegexFilter::EXTRA_WORDS,
            RegexFilter::EXTRA_WHITESPACE
        ])
    ], new MultilayerPerceptron([
        new Dense($nodeSize), new Activation(new LeakyReLU()),
        new Dense($nodeSize), new Activation(new LeakyReLU()),
        new Dense($nodeSize, $alphaSize, false), // false bias
        new BatchNorm(),
        new Activation(new LeakyReLU()),
        new Dense($nodeSize / 2), new PReLU(), // half node
        new Dense($nodeSize / 2), new PReLU(),
    ], $batchSize, new AdaMax())), // 0.0001 learning rate
    new Filesystem($rubixModelPath, true)
);

/* $estimator->setBackend(new Amp(4)); */
$estimator->setLogger($logger);
$estimator->train($dataset);
$extractor = new CSV($progresDataPath, true);
$extractor->export($estimator->steps());

$logger->info('Progress saved to progress.csv');

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $estimator->save();
}
