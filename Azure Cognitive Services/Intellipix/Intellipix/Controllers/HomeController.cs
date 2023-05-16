using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Intellipix.Models;
using System.Drawing;
using System.IO;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Hosting;
using Azure.Storage.Blobs;
using Microsoft.Extensions.Configuration;
using Azure.Storage.Blobs.Models;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision;
using Microsoft.Azure.CognitiveServices.Vision.ComputerVision.Models;

namespace Intellipix.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly IConfiguration _configuration;

        public HomeController(ILogger<HomeController> logger, IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        }

        public async Task<IActionResult> Index(string term)
        {
            BlobServiceClient client = new BlobServiceClient(_configuration.GetConnectionString("Storage"));
            BlobContainerClient container = client.GetBlobContainerClient("photos");
            List<BlobData> blobs = new List<BlobData>();
            term = term?.Trim();

            await foreach (BlobItem item in container.GetBlobsAsync(BlobTraits.Metadata))
            {
                if (String.IsNullOrEmpty(term) ||
                    item.Metadata["Caption"].Contains(term, StringComparison.CurrentCultureIgnoreCase) ||
                    item.Metadata["Tags"].Contains(term, StringComparison.CurrentCultureIgnoreCase))
                {
                    BlobClient blob = container.GetBlobClient(item.Name);

                    blobs.Add(new BlobData()
                    {
                        ImageUri = blob.Uri.ToString(),
                        ThumbnailUri = blob.Uri.ToString().Replace("/photos/", "/thumbnails/"),
                        Caption = item.Metadata["Caption"]
                    });
                }
            }

            ViewBag.Blobs = blobs.ToArray();
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }

        [HttpPost]
        public async Task<ActionResult> Upload(IFormFile file)
        {
            if (file?.Length > 0)
            {
                // Make sure the user selected an image file
                if (!file.ContentType.StartsWith("image"))
                {
                    TempData["Message"] = "Only image files may be uploaded";
                }
                else
                {
                    try
                    {
                        // Save the original image in the "photos" container
                        BlobServiceClient client = new BlobServiceClient(_configuration.GetConnectionString("Storage"));
                        BlobContainerClient container = client.GetBlobContainerClient("photos");
                        BlobClient photo = container.GetBlobClient(Path.GetFileName(file.FileName));
                        await photo.UploadAsync(file.OpenReadStream(), true);

                        // Generate a thumbnail and save it in the "thumbnails" container
                        using (var formFileStream = file.OpenReadStream())

                        using (var sourceImage = Image.FromStream(formFileStream))
                        {
                            var newWidth = 192;
                            var newHeight = (Int32)(1.0 * sourceImage.Height / sourceImage.Width * newWidth);
                            using (var destinationImage = new Bitmap(sourceImage, new Size(newWidth, newHeight)))

                            using (var stream = new MemoryStream())
                            {
                                destinationImage.Save(stream, sourceImage.RawFormat);
                                stream.Seek(0L, SeekOrigin.Begin);
                                container = client.GetBlobContainerClient("thumbnails");
                                var thumbnail = container.GetBlobClient(Path.GetFileName(file.FileName));
                                await thumbnail.UploadAsync(stream, true);
                            }
                        }

                        // Submit the image to the Computer Vision API
                        ComputerVisionClient vision = new ComputerVisionClient(
                            new ApiKeyServiceClientCredentials(_configuration.GetSection("Vision").GetValue<String>("Key")),
                            new System.Net.Http.DelegatingHandler[] { }
                        );

                        vision.Endpoint = _configuration.GetSection("Vision").GetValue<String>("Endpoint");
                        var features = new List<VisualFeatureTypes?>() { VisualFeatureTypes.Description };
                        var result = await vision.AnalyzeImageAsync(photo.Uri.ToString(), features);

                        // Record the image caption and tags in blob metadata
                        var metadata = new Dictionary<string, string>()
                        {
                            { "Caption", result.Description.Captions[0].Text },
                            { "Tags", String.Join(';', result.Description.Tags.ToArray()) }
                        };

                        await photo.SetMetadataAsync(metadata);
                    }
                    catch (Exception ex)
                    {
                        // In case something goes wrong
                        TempData["Message"] = ex.Message;
                    }
                }
            }

            return RedirectToAction("Index");
        }

        [HttpPost]
        public ActionResult Search(string term)
        {
            return RedirectToAction("Index", new { term = term });
        }
    }
}
