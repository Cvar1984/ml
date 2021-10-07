<?php

require './vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\PersistentModel;

/* $samples = [ */
/*     [3, 4, 50.5], */
/*     [1, 5, 24.7], */
/*     [4, 4, 62.0], */
/*     [3, 2, 31.1], */
/* ]; */

/* $labels = ['married', 'divorced', 'married', 'divorced']; */
/* $dataset = new Labeled($samples, $labels); */
$dataset = Labeled::fromIterator(
    new CSV('assets/dataset/marriage.csv', true)
)->apply(new NumericStringConverter());
/* echo $dataset->numSamples(); */

$estimator = new PersistentModel(
    new KNearestNeighbors(3),
    new Filesystem('assets/rubix/marriage.rbx')
);
/* $validator = new HoldOut(0.2); // 20% test 80% train */
/* $score = $validator->test($estimator, $dataset, new Accuracy()); */
/* echo $score; */
$estimator->train($dataset);

if($estimator->trained())
{
    if(readline('save this model? [Y/N]') == 'y') {
        $estimator->save();
    }
}

