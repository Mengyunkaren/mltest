# MLTest

1.Enter the directory and run `sbt package`. The jar file will be generated at `/target/scala-2.11/cerebri_test_2.11-0.1.jar`

2.Run 
```bash
spark-submit \
  --class "RandomForest" \
  --master local[4] \
  target/scala-2.11/cerebri_test_2.11-0.1.jar
```


3. The parquet output will be written to `ml_test`

4. For modeling, the column called `name` was dropped

5. The final scores obtained are as following:
```txt
Precision: 0.8709677419354839
Recall: 0.9642857142857143
AUC: 0.759920634920635
```

