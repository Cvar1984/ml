<?php

$array = $fields = array(); $i = 0;
$handle = @fopen('assets/dataset/sentiment.csv', "r");
if ($handle) {
    while (($row = fgetcsv($handle, 1024, ',')) !== false) {
        if (empty($fields)) {
            $fields = $row;
            continue;
        }
        foreach ($row as $k=>$value) {
            $array[$i][$fields[$k]] = $value;
        }
        $i++;
    }
    if (!feof($handle)) {
        echo "Error: unexpected fgets() fail\n";
    }
    fclose($handle);
}
$size = sizeof($array);
$counterA = $counterB = 0;
for($x = 0; $x < $size; $x++){
    if($array[$x]['label'] == 1 &&
        !empty($array[$x]['review_sangat_singkat'])) {

        file_put_contents(
            'positive/'.$counterA.'.txt',
            $array[$x]['review_sangat_singkat']
        );

        $counterA++;

    } elseif($array[$x]['label'] == 0 &&
        !empty($array[$x]['review_sangat_singkat'])) {

        file_put_contents(
            'negative/'.$counterB.'.txt',
            $array[$x]['review_sangat_singkat']
        );

        $counterB++;
    } 
}
/* print_r($array); */
