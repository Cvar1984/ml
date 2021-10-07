<?php

require './vendor/autoload.php';

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\Transformers\NumericStringConverter;

/* $samples = [ */
/*     [4, 3, 44.2], */
/*     [2, 2, 16.7], */
/*     [2, 4, 19.5], */
/*     [3, 3, 55.0], */
/* ]; */

/* $dataset = new Unlabeled($samples); */
$samples = new CSV('assets/dataset/marriage.csv', true);
$dataset = Labeled::fromIterator($samples)
    ->apply(new NumericStringConverter());
$estimator = PersistentModel::load(
    new Filesystem('assets/rubix/marriage.rbx')
);

$predictions = $estimator->predict($dataset);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $dataset->labels());
$results->toJSON()->saveTo(new Filesystem('assets/report/marriage.json'));
print_r($predictions);
