#include<opencv2\opencv.hpp>
#include<vector>
#define len 440
#define wid 140

using namespace cv;

float distance(Point2f p1, Point2f p2);

int main(void)
{
	Mat src1, src2, src3, src4,src5;
	std::vector<std::vector<Point>>contours, contours2, temp;
	std::vector<Vec4i>hierarchy, hierarchy2;
	Mat src = cv::imread("../test.png");
	src2 = src.clone();
	std::vector<Mat> src_split;
	if (src.empty()) {
		std::cout << "测试图片路径错误";
		return-1;
	}
	imshow("src", src);

	split(src, src_split);
	src = src_split[0] * 2 - src_split[1] - src_split[2];//分离蓝色空间
	threshold(src, src, 100, 255, THRESH_BINARY);//二值化


	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	Point2f points[4], points2[4];
	int n = contours.size() - 1;//最大轮廓


	std::vector<Point2f>result;
	for (double i = 0;; i += 0.01)
	{
		approxPolyDP(contours[n], result, i, true);//点顺序左上左下右下 最小外接多边形
		if (result.size() == 4)//限定边数为4
			break;
	}

	//RotatedRect rrect = minAreaRect(contours[n]);//最小外接矩形
	//Point2f cpt = rrect.center;
	//rrect.points(points);//左下,左上,右上
	//for (int i = 0; i < result.size(); ++i) {
	//	if (i == result.size() -1)
	//		line(src2, result[i], result[0], Scalar(0, 0, 255), 2);
	//	else
	//		line(src2, result[i], result[i+1], Scalar(0, 0, 255), 2);
	//}
	//circle(src2, cpt, 4, Scalar(0, 255, 0), -1);

	/*Mat mask = Mat::zeros(src.size(), CV_8UC3);//去除外部
	for (int i = 0; i < 4; ++i) {
		if (i == 3)
			line(mask, result[i], result[0], Scalar(255, 255, 255), 2);
		else
			line(mask, result[i], result[i + 1], Scalar(255, 255, 255), 2);
	}
	floodFill(mask, cpt, Scalar(255,255,255));
	src2 = src2 & mask;*/
	Point2f dstpts[4] = { Point2f(0,0),Point2f(0,wid),Point2f(len,wid),Point2f(len,0) };
	for (int i = 0; i < 4; ++i) points2[i] = result[i];
	Mat rotation = getPerspectiveTransform(points2, dstpts);
	warpPerspective(src2, src2, rotation, Size(len, wid));//映射到440*140
	src3 = src2.clone();
	imshow("src2", src2);
	cvtColor(src2, src2, COLOR_BGR2GRAY);
	threshold(src2, src2, 150, 255, THRESH_BINARY);//二值化

	Mat maskk=Mat::zeros(src2.size(), CV_8UC1);
	rectangle(maskk, Rect(0, 0, 55, 140), Scalar(255), -1);//取出汉字单独处理
	src4 = src2 & maskk;

	//Mat kernel2 = getStructuringElement(MORPH_RECT, Size(3,3));
	//morphologyEx(src2, src2, MORPH_OPEN, kernel2,Point(-1,-1),2);

	findContours(src2, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours2.size(); ++i)
	{
		if (contourArea(contours2[i]) <600)//限定大小去除螺钉
			temp.push_back(contours2[i]);
	}
	drawContours(src2, temp, -1, Scalar(0), -1);

	src2 = src2 | src4;//送回汉字
	src5 = src2.clone();

	Mat kernel1 = getStructuringElement(MORPH_RECT, Size(15,15));
	dilate(src4, src4, kernel1);
	src2 = src2 | src4;//送回汉字

	src4 = Mat::zeros(src2.size(), CV_8UC3);
	findContours(src2, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int j = 0; j < contours2.size(); ++j)
	{
		RotatedRect rrect = minAreaRect(contours2[j]);
		Point2f points2[4];
		rrect.points(points2);
		if (distance(points2[0], points2[1]) * distance(points2[2], points2[1]) > 1000)
		{
			for (int i = 0; i < 4; ++i) {
				if (i == 3)
					line(src4, points2[i], points2[0], Scalar(255,255,255), 1);
				else
					line(src4, points2[i], points2[i + 1], Scalar(255,255,255), 1);
			}
		}
	}
	floodFill(src4, Point(0, 0), Scalar(255,255,255));
	src3 &= ~src4;
	imshow("src3", src3);
	Mat tem[3];
	split(src4, tem);
	src5 = ~tem[0]&~src5;
	imshow("src5", src5);
	waitKey();
	return 0;
}

float distance(Point2f p1, Point2f p2)
{
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}