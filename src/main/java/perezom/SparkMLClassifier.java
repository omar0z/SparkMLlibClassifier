package perezom;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class SparkMLClassifier {

    public static void main(String[] args) {

        // Create Spark context
        var session = getSession();

        // Load the data
        var data = getDataset(session);
        var features = data.columns();

        // Explore the data
        var summaryData = data.describe();
        System.out.println("Summary: \n");
        summaryData.show();

        // Get the Train/Test splits
        var splits = getTrainTestSplit(data);
        var trainingData = splits[0];
        var testData = splits[1];


        // Model the data

        var assembler = new VectorAssembler()
                .setInputCols(new String[] {"sepalLength", "sepalWidth", "petalLength", "petalWidth"})
                .setOutputCol("raw");

        var featureIndexer = new VectorIndexer()
                .setInputCol("raw")
                .setOutputCol("features")
                .setMaxCategories(4);

        var labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("label");

        RandomForestClassifier randomForestClassifier = new RandomForestClassifier()
                .setFeaturesCol("features")
                .setLabelCol("label");

        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[]{
                        assembler
                        , featureIndexer
                        , labelIndexer
                        , randomForestClassifier});
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions
        var predictions = model.transform(testData);
        predictions.select("prediction", "features", "label").show(10);

        // Evaluate the model
        var evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));


        RandomForestClassificationModel rfModel = (RandomForestClassificationModel) (model.stages()[3]);
        System.out.println("Learned classification forest model:\n" + rfModel.toDebugString());

        //ggwp
    }

    public static SparkSession getSession() {
        SparkSession session = SparkSession.builder()
                .appName("SparkMLClassifier")
                .config("spark.master", "local[8]")
                .getOrCreate();
        return session;
    }

    public static Dataset<Row> getDataset(SparkSession session) {
        Dataset<Row> irisData = session.read().format("csv")
                .option("sep", ",")
                .option("inferSchema", "true")
                .option("header", "true")
                .load("./src/main/resources/iris.csv");

        return irisData;
    }

    public static Dataset<Row>[] getTrainTestSplit(Dataset<Row> data) {
        return data.randomSplit(new double[]{0.8, 0.2});
    }
}
