<?php

require './vendor/autoload.php';
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Tokenizers\NGram;
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
/* use Rubix\ML\Backends\Amp; */
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

$estimator = new PersistentModel(
    new Pipeline([
        new TextNormalizer(),
        new WordCountVectorizer(10000, 0.00008, 0.4, new NGram(1, 2)),
        new TfIdfTransformer(),
        new ZScaleStandardizer(),
    ], new MultilayerPerceptron([
        new Dense(100),
        new Activation(new LeakyReLU()),
        new Dense(100),
        new Activation(new LeakyReLU()),
        new Dense(100, 0.0, false),
        new BatchNorm(),
        new Activation(new LeakyReLU()),
        new Dense(50),
        new PReLU(),
        new Dense(50),
        new PReLU(),
    ], 256, new AdaMax(0.0001))),
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
