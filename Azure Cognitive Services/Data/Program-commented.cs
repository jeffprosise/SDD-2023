// Import necessary libraries
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace SentimentAnalysis
{
    class Program
    {
        // Define the paths to load and save the data
        static readonly string _loadPath = "..\\..\\..\\Data\\yelp_labelled.tsv";
        static readonly string _savePath = "..\\..\\..\\Data\\Sentiment.zip";

        // Main function
        static void Main(string[] args)
        {
            // Create MLContext
            var context = new MLContext(seed: 0);

            // Load data from the input file
            var data = context.Data.LoadFromTextFile<Input>(_loadPath, hasHeader: false);

            // Split the data into train and test datasets
            var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Build the pipeline
            var pipeline = context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "SentimentText")
                .Append(context.BinaryClassification.Trainers.FastTree(numberOfLeaves: 50, minimumExampleCountPerLeaf: 20));

            // Train the model
            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            // Use the model to make predictions on test data and evaluate the performance
            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            // Print evaluation metrics
            Console.WriteLine();
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"F1: {metrics.F1Score:P2}");
            Console.WriteLine();

            // Save the trained model
            Console.WriteLine("Saving the model...");
            context.Model.Save(model, data.Schema, _savePath);
        }
    }

    // Define input class structure
    public class Input
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    // Define output class structure
    public class Output
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}