// Import necessary namespaces
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

// Define namespace and class
namespace SentimentAnalysis
{
    class Program
    {
        // Define paths for loading and saving data
        static readonly string _loadPath = "..\\..\\..\\Data\\yelp_labelled.tsv";
        static readonly string _savePath = "..\\..\\..\\Data\\Sentiment.zip";

        // Main method for program execution
        static void Main(string[] args)
        {
            // Create new MLContext object for machine learning operations
            var context = new MLContext(seed: 0);

            // Load data from file into Input object
            var data = context.Data.LoadFromTextFile<Input>(_loadPath, hasHeader: false);

            // Split data into test and train datasets
            var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Define machine learning pipeline for sentiment analysis
            var pipeline = context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "SentimentText")
                .Append(context.BinaryClassification.Trainers.FastTree(numberOfLeaves: 50, minimumExampleCountPerLeaf: 20));

            // Train model using training dataset
            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            // Test model using testing dataset
            var predictions = model.Transform(testData);

            // Evaluate model performance
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            // Print evaluation results
            Console.WriteLine();
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"F1: {metrics.F1Score:P2}");
            Console.WriteLine();

            // Save trained model for future use
            Console.WriteLine("Saving the model...");
            context.Model.Save(model, data.Schema, _savePath);
        }
    }

    // Define Input and Output classes for loading and processing data
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