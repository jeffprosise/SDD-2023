using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _loadPath = "..\\..\\..\\Data\\yelp_labelled.tsv";
        static readonly string _savePath = "..\\..\\..\\Data\\Sentiment.zip";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);
            var data = context.Data.LoadFromTextFile<Input>(_loadPath, hasHeader: false);

            var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            var pipeline = context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "SentimentText")
                .Append(context.BinaryClassification.Trainers.FastTree(numberOfLeaves: 50, minimumExampleCountPerLeaf: 20));

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"F1: {metrics.F1Score:P2}");
            Console.WriteLine();

            Console.WriteLine("Saving the model...");
            context.Model.Save(model, data.Schema, _savePath);
        }
    }

    public class Input
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    public class Output
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}