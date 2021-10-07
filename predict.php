<?php

require './vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Loggers\Screen;

ini_set('memory_limit', '-1');

$logger = new Screen();

$estimator = PersistentModel::load(
    new Filesystem('assets/rubix/sentiment.rbx')
);

while (empty($text)) $text = readline("Enter some text to analyze:\n");

$dataset = new Unlabeled([[$text]]);
$prediction = current($estimator->predict($dataset));

$logger->info('The sentiment is: ' . $prediction);
