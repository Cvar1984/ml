<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use League\Csv\Reader;
use League\Csv\Statement;

ini_set('memory_limit', '-1');

$rubixModelPath = 'assets/rubix/sentiment.rbx';
$reportDataPath = 'assets/report/sentiment.json';
$labeledDataPath = 'assets/dataset/sentiment.csv';

// WARNING you should avoid using training dataset for testing process
$csv = Reader::createFromPath($labeledDataPath);
$csv->setDelimiter(',');
$csv->setHeaderOffset(0);

//build a statement
$stmt = Statement::create()
    ->offset(0)
    ->limit($argv[1]); // limit rows to be loaded

//query your records from the document
$records = $stmt->process($csv);

$samples = $labels = [];
foreach ($records as $record) {
    $samples[] = [$record['review_sangat_singkat']];
    $labels[] = $record['label'];
}

$logger = new Screen();
$logger->info('Loading data into memory');

$dataset = Labeled::build($samples, $labels)->randomize()->take(96328);

$estimator = PersistentModel::load(new Filesystem($rubixModelPath));

$logger->info('Making predictions');

$predictions = $estimator->predict($dataset);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $dataset->labels());
$logger->info($results);
$results
    ->toJSON()
    ->saveTo(new Filesystem($reportDataPath));

$logger->info('Report saved');
