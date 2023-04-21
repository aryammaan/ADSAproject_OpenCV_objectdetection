#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define THRESHOLD_VAlUE 90
#define KERNEL_SIZE 3
#define SIGMA 1

void grayscale(Mat &image, Mat &gray_image) {
   // Loop through each pixel in the input image and convert to grayscale
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {

            //vec3b is the channel to access the r g b values of the pixel
            Vec3b pixel = image.at<Vec3b>(i, j);

            //multiplying with the luma coefficients to convert each pixel into grayscale
            uchar gray_value = 0.2126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0];

            //uchar is the function to access the grayscale value of the pixel
            gray_image.at<uchar>(i, j) = gray_value;
        }
    }
}

// void threshold(Mat &image, Mat &thresholded_image, int threshold_value){

    
//      for (int i = 0; i < image.rows; i++) {
//         for (int j = 0; j < image.cols; j++) {
//             if (image.at<uchar>(i, j) > threshold_value) {
//                 thresholded_image.at<uchar>(i, j) = 0;
//             }
//             else {
//                 thresholded_image.at<uchar>(i, j) = 255;
//             }
//         }
//     }
// }

void threshold(Mat &image, Mat &thresholded_image) 
{

    // Calculate histogram
    int hist[256] = {0};
    int num_pixels = image.rows * image.cols;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int intensity = (int)image.at<uchar>(i, j);
            hist[intensity]++;
        }
    }

    // Normalize histogram
    double norm_hist[256] = {0};
    for (int i = 0; i < 256; i++) {
        norm_hist[i] = (double)hist[i] / num_pixels;
    }

    // Calculate cumulative sums
    double cum_sum[256] = {0};
    cum_sum[0] = norm_hist[0];
    for (int i = 1; i < 256; i++) {
        cum_sum[i] = cum_sum[i-1] + norm_hist[i];
    }

    // Calculate global mean intensity
    double global_mean = 0;
    for (int i = 0; i < 256; i++) {
        global_mean += i * norm_hist[i];
    }

    // Calculate inter-class variance for each threshold
    double max_variance = 0;
    int threshold = 0;
    for (int i = 0; i < 256; i++) {
        double w0 = cum_sum[i];
        double w1 = 1 - w0;
        double mean0 = 0;
        double mean1 = 0;
        for (int j = 0; j <= i; j++) {
            mean0 += j * norm_hist[j] / w0;
        }
        for (int j = i+1; j < 256; j++) {
            mean1 += j * norm_hist[j] / w1;
        }
        double variance = w0 * w1 * pow((mean0 - mean1), 2);
        if (variance > max_variance) {
            max_variance = variance;
            threshold = i;
        }
    }

    // Threshold the image
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) > threshold) {
                thresholded_image.at<uchar>(i, j) = 0;
            }
            else {
                thresholded_image.at<uchar>(i, j) = 255;
            }
        }
    }
}


void gaussianBlur(const cv::Mat& input, cv::Mat& output, int kernel_size, double sigma)
{
    int k = (kernel_size - 1) / 2;
    double kernel[kernel_size][kernel_size];
    double sum = 0.0;

    // Create Gaussian kernel
    for (int x = -k; x <= k; x++) {
        for (int y = -k; y <= k; y++) {
            double value = exp(-(x*x + y*y) / (2.0*sigma*sigma));
            kernel[x + k][y + k] = value;
            sum += value;
        }
    }

    // Normalize kernel
    for (int x = 0; x < kernel_size; x++) {
        for (int y = 0; y < kernel_size; y++) {
            kernel[x][y] /= sum;
        }
    }

    // Apply convolution
    output.create(input.size(), input.type());
    for (int i = k; i < input.rows - k; i++) {
        for (int j = k; j < input.cols - k; j++) {
            double sum = 0.0;
            for (int x = -k; x <= k; x++) {
                for (int y = -k; y <= k; y++) {
                    sum += kernel[x + k][y + k] * input.at<uchar>(i + x, j + y);
                }
            }
            output.at<uchar>(i, j) = sum;
        }
    }
}

// void detectEdges(Mat& image /*, vector<vector<Point>>& contours*/)
// {
//     //duplicating image
//     cv::Mat new_img(image.size(), image.type());
//     image.copyTo(new_img);

//     // Find contours in the binary image
//     std::vector<std::vector<cv::Point>> contours;
//     cv::findContours(new_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

//     // Draw the contours as a series of connected lines using cv::line
//     Mat drawing = Mat::zeros(image.size(), CV_8UC3);
//     cv::Scalar color(0, 255, 0);  // Green color
//     int thickness = 1;
//     // drawing lines between each consecutive points in the counter
//     for (size_t i = 0; i < contours.size(); i++) {
//         for (size_t j = 0; j < contours[i].size(); j++) {
//             Point p1 = contours[i][j];
//             Point p2;
//             if (j < contours[i].size() - 1) {
//                 p2 = contours[i][j + 1];
//             } else {
//                 p2 = contours[i][0];
//             }
//             line(drawing, p1, p2, color, thickness);
//         }
//     }

//     // Display the image with the contours
//     imwrite("processing/contours.jpg", drawing);

// }

void detectEdges(Mat& image /*, vector<vector<Point>>& contours*/)
{
    // Apply the Sobel operator to the input image
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    cv::Sobel(image, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(image, grad_y, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    Mat grad;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // Find contours in the gradient image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(grad, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw the contours on the gradient image
    Mat drawing = Mat::zeros(grad.size(), CV_8UC3);
    cv::Scalar color(0, 255, 0);  // Green color
    int thickness = 1;
    for (size_t i = 0; i < contours.size(); i++) {
        for (size_t j = 0; j < contours[i].size(); j++) {
            Point p1 = contours[i][j];
            Point p2;
            if (j < contours[i].size() - 1) {
                p2 = contours[i][j + 1];
            } else {
                p2 = contours[i][0];
            }
            line(drawing, p1, p2, color, thickness);
        }
    }

    // Display the image with the contours
    imwrite("processing/contours2.jpg", drawing);
}


int main()
{
    // Load the image
    Mat image = imread("../images/football1.jpeg");


    // Create a new grayscale image with the same size as the input image
    //CV_8UC1 - cv image with 8 bit unsigned with only one channel (ie grayscale)
    Mat gray_image(image.size(), CV_8UC1);

    grayscale(image, gray_image);
 
    // Save the gray image
    imwrite("processing/gray_image_function.jpg", gray_image);


    // create a one channel image of the same size as image
    Mat thresholded_image(image.size(), CV_8UC1);
    
    //127 being in the middle of 0 to 255, so generally we threshold using that
    // threshold(gray_image, thresholded_image, THRESHOLD_VAlUE);
    threshold(gray_image, thresholded_image);
    imwrite("processing/thresholded_image1.jpg", thresholded_image);

    // Apply Gaussian blur
    Mat blurred_image(image.size(), CV_8UC1);
    gaussianBlur(thresholded_image, blurred_image, KERNEL_SIZE, SIGMA);
    // GaussianBlur(thresholded_image, blurred_image, Size(3,3), 0);
    imwrite("processing/processed_image2.jpg", blurred_image);

    Mat binary_image(image.size(), CV_8UC1);
    
    //127 being in the middle of 0 to 255, so generally we threshold using that
    // threshold(blurred_image, binary_image, THRESHOLD_VAlUE);
     threshold(blurred_image, binary_image);
    imwrite("processing/thresholded_image2.jpg", thresholded_image);

    vector<vector<Point>> contours;
    detectEdges(blurred_image /*, contours */);
 
   
   

    return 0;
}