<?php

require './vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Loggers\Screen;

ini_set('memory_limit', '-1');

$logger = new Screen();

$rubixDataPath = 'assets/rubix/sentiment.rbx';

$estimator = PersistentModel::load(new Filesystem($rubixDataPath));

while (empty($text)) $text = readline("Enter some text to analyze:\n");

$dataset = new Unlabeled([[$text]]);
$prediction = current($estimator->predict($dataset));
$prediction = ($prediction > 0) ? 'positive' : 'negative';

$logger->info('The sentiment is: ' . $prediction);
