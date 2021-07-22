#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
	cv::Mat image;

	if (argc != 2) {
		std::fprintf(stderr, "usage: %s IMAGEPATH\n", *argv);
		return EXIT_FAILURE;
	}

	image = cv::imread(argv[1], 1);
	if (!image.data) {
		std::fprintf(stderr, "Couldn't read image file\n");
		return EXIT_FAILURE;
	}
	cv::namedWindow("mama mia", cv::WINDOW_AUTOSIZE);
	cv::imshow("mama mia", image);

	cv::waitKey(0);
}
