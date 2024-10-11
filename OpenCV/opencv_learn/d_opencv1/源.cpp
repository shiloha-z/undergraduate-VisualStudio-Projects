#include <opencv2/opencv.hpp>
#include<cmath>
#include<vector>
using namespace cv;

auto anotherAddGaussianNoise(Mat&)->Mat;
auto add_salt_pepper_noise(Mat&)->Mat;
auto hist_draw(Mat&)->void;
auto my_dft(Mat, Mat&, int)->void;
Mat max_min_functin(Mat img, int ksize, int flag);
Mat center_point(Mat img, int ksize);//�е��˲���
Mat pow_blur(Mat img, int ksize);//���ξ�ֵ�˲� ksize���ܹ���
Mat HomoFilter(Mat inputImg);
Mat YHomoFilter(Mat orginalImg);
double getSNR(Mat& srcImage, Mat& dstImage);
Mat cyclical_noise(Mat img);
void genaratePsf(Mat& psf, cv::Point& anchor, double len, double angle);
void Shifting_DFT(Mat& fImage);//Ƶ�װ���
Mat checker_generator(void);//���̸�����
Mat GetSpectrum(const Mat& src);
Mat WienerFilter(const Mat& src, const Mat& ref, int stddev);

int main(void)
{
	Mat src1, src2, temp1, temp2, dst1, dst2, mdst1, mdst2, salt, gauss, cyc, kernel, motion_blur;
	Point p1;
	Mat src = imread("./lena512.bmp");
	Mat img1 = imread("./1.png");
	Mat img2 = imread("./2.png");

	src1 = src.clone();
	salt = add_salt_pepper_noise(src);
	cvtColor(salt, salt, COLOR_BGR2GRAY);
	cvtColor(src, src, COLOR_BGR2GRAY);
	cyc = cyclical_noise(src);
	gauss = anotherAddGaussianNoise(src);
	src2 = Mat_<float>(src);
	genaratePsf(kernel, p1, 50, 10);
	filter2D(src, motion_blur, -1, kernel, p1);
	{//sobel���ӱ�Եǿ��
		//Sobel(src, dst1, -1, 1, 1, 3, 1);
		//addWeighted(src, 1, dst1, -1, 0, temp2);
		////imshow("sobel0", dst1);
		//imshow("sobel", temp2);
	}
	{//��˹��ͨ�˲�
		//my_dft(src2, mdst2, 2);
		//normalize(mdst2, mdst2, 0, 1, NORM_MINMAX);
		//normalize(src2, src2, 0, 1, NORM_MINMAX);
		//addWeighted(src2, 1, mdst2, -1, 0, mdst1);
		////normalize(mdst1, mdst1, 0, 1, NORM_MINMAX);
		//imshow("gauss", mdst1);
		//imshow("src", src);
	}
	{//��ֵ�˲�
		//cv::imshow("GaussianNoise", gauss);
		//medianBlur(gauss, mdst1, 3);
		//cv::imshow("GaussianMedianBlur", mdst1);
		//std::cout << "\n��˹��ֵ�˲������:" << getSNR(gauss, mdst1);
		//blur(gauss, mdst2, Size(5, 5));
		//cv::imshow("��ֵ�˲�", mdst2);
		//std::cout<<"\n��ֵ�˲������:"<<getSNR(gauss, mdst2);
	}
	{//��˹�˲�
		//GaussianBlur(gauss, mdst1, Size(3, 3), 0);
		//cv::imshow("��˹�˲�", mdst2);
	}
	{// ����Ҷ��ͨ�˲�
		//temp1 = Mat_<float>(gauss);
		//my_dft(temp1, mdst2, 1);
		//cv::imshow("��ͨ�˲�", mdst2);
		//std::cout << "\n����Ҷ��ͨ�˲������:" << getSNR(gauss, mdst2);
	}
	{//�е��˲���
		//imshow("gauss", gauss);
		//mdst1 = center_point(gauss, 3);
		//imshow("�е��˲���", mdst1);
	}
	{//���ξ�ֵ�˲�
		//imshow("gauss", gauss);
		//dst1 = pow_blur(gauss, 3);
		//imshow("���ξ�ֵ�˲���", dst1);
	}
	{//̬ͬ�˲���
		//imshow("src", src1);
		//dst1 = YHomoFilter(src1);
		//imshow("̬ͬ�˲���", dst1);
	}
	{//��������ȥ��
		my_dft(cyc, dst1, 3);
	}
	//dst1 = checker_generator();
	//imshow("img1", img1);
	//imshow("img2", img2);
	//Mat kernel1= getStructuringElement(MORPH_CROSS, Size(3, 3));
	//dilate(img1, img1, kernel1);
	//imshow("dilate", img1);
	//Mat kernel2 = getStructuringElement(MORPH_RECT, Size(15, 15));
	//erode(img2, img2, kernel2,Point(-1,-1),3);
	//imshow("erode", img2);
	//dst1=WienerFilter(gauss, src, 20);
	imshow("c", cyc);
	imshow("s", dst1);

	waitKey();
}

auto anotherAddGaussianNoise(Mat& srcImg)->Mat
{
	Mat tempSrcImg = srcImg.clone();
	//�����˹��������
	Mat noise(tempSrcImg.size(), tempSrcImg.type());//����һ����������
	RNG rng(time(NULL));
	rng.fill(noise, RNG::NORMAL, 10, 36);//��˹�ֲ�����ֵΪ10����׼��Ϊ36
	//����˹����������ԭͼ����ӵõ�����ͼ��
	cv::add(tempSrcImg, noise, tempSrcImg);
	return tempSrcImg;
}
auto add_salt_pepper_noise(Mat& image)->Mat
{
	Mat temp_salt_img = image.clone();
	RNG rng(time(NULL));
	for (int i = 0; i < 10000; i++) {
		int x = rng.uniform(0, temp_salt_img.cols);
		int y = rng.uniform(0, temp_salt_img.rows);
		if (i % 2 == 1)
			temp_salt_img.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
		else
			temp_salt_img.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
	}
	return temp_salt_img;
}
auto hist_draw(Mat& img)->void
{
	if (img.channels() != 1)//�Ҷ�ͼת��
		cvtColor(img, img, COLOR_RGB2GRAY);
	Mat hist;
	const int hist_size = 256;
	const float range[] = { 0,256 };
	const float* histRange = range;
	calcHist(&img, 1, 0, Mat(), hist, 1, &hist_size, &histRange);//hist col=1,row=256,hist val=������Ŀ
	int hist_h = 400;//ֱ��ͼ�߶�
	int hist_w = 512;//ֱ��ͼ���
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));//ֱ��ͼ����ɫ
	normalize(hist, hist, 0, hist_h, NORM_MINMAX, -1, Mat());//
	for (int i = 0; i < hist_size; ++i)
	{
		float bin_val = hist.at<float>(i);
		line(histImage, Point(i * 2, 400), Point(i * 2, (400 - bin_val)), Scalar(255, 0, 0), 1, LINE_AA);//ֱ�߻���
	}
	imshow("ֱ��ͼ", histImage);
}
auto my_dft(Mat input, Mat& output, int mode)->void
{
	Mat flo[2], complex, temp1, temp2;

	int row = getOptimalDFTSize(input.rows);
	int col = getOptimalDFTSize(input.cols);
	int T = (row - input.rows) / 2, B = (row - input.rows) / 2, L = (col - input.cols) / 2, R = (col - input.cols) / 2;
	copyMakeBorder(input, temp1, T, B, L, R, BORDER_CONSTANT);

	flo[0] = Mat_<float>(temp1);
	flo[1] = Mat::zeros(temp1.size(), CV_32F);
	merge(flo, 2, complex);
	dft(complex, temp2);

	Mat resultC[2];
	Mat	amagnitude;
	split(temp2, resultC);
	magnitude(resultC[0], resultC[1], amagnitude);

	//amagnitude += Scalar(1);
	log(amagnitude, amagnitude);
	amagnitude = amagnitude(Rect(T, L, input.cols, input.rows));
	normalize(amagnitude, amagnitude, 0, 1, NORM_MINMAX);

	Shifting_DFT(amagnitude);
	imshow("amagnitude", amagnitude);
	Shifting_DFT(resultC[0]);
	Shifting_DFT(resultC[1]);
	int center_x = resultC[0].rows / 2;
	int center_y = resultC[0].cols / 2;
	if (mode == 1)//��ͨ�˲�
	{
		int r = 30;//ԽС��ƵԽ��
		float h, n = 2, d;
		for (int i = 0; i < resultC[0].rows; ++i)
		{//������˹��ͨ�˲�
			for (int j = 0; j < resultC[0].cols; ++j)
			{
				d = (i - center_x) * (i - center_x) + (j - center_y) * (j - center_y);
				h = 1 / (1 + pow((d / (r * r)), n));
				resultC[0].at<float>(i, j) *= h;
				resultC[1].at<float>(i, j) *= h;
			}
		}
		//{//�����ͨ
		//	Mat te = Mat::zeros(resultC[0].size(), resultC[0].type());
		//	circle(te, Point(te.rows / 2, te.cols / 2),80, 1, -1);
		//	multiply(resultC[0], te, resultC[0]);
		//	multiply(resultC[1], te, resultC[1]);
		//}
	}
	else if (mode == 2)//��ͨ�˲�
	{
		int r = 1200;//Խ���ƵԽ��
		float h, n = 2, d;
		for (int i = 0; i < resultC[0].rows; ++i)
		{//������˹��ͨ�˲�
			for (int j = 0; j < resultC[0].cols; ++j)
			{
				d = (i - center_x) * (i - center_x) + (j - center_y) * (j - center_y);
				h = 1 / (1 + pow(((r * r) / d), n));
				resultC[0].at<float>(i, j) *= h;
				resultC[1].at<float>(i, j) *= h;
			}
		}
		{//�����ͨ�˲�
			//circle(resultC[0], Point(amagnitude.rows / 2, amagnitude.cols / 2), 50, 0, -1);
			//circle(resultC[1], Point(amagnitude.rows / 2, amagnitude.cols / 2), 50, 0, -1);
		}
	}
	else if (mode == 3)//�����ݲ��˲���
	{
		circle(resultC[0], Point(center_x, 234), 3, Scalar(0), -1);
		circle(resultC[0], Point(center_x, 278), 3, Scalar(0), -1);
		circle(resultC[1], Point(center_x, 234), 3, Scalar(0), -1);
		circle(resultC[1], Point(center_x, 278), 3, Scalar(0), -1);
	}
	else if (mode == 4)//�����˲�
	{
		Mat te = Mat::ones(resultC[0].size(), resultC[0].type());
		circle(te, Point(te.rows / 2, te.cols / 2), 27, 0, -1);
		circle(te, Point(te.rows / 2, te.cols / 2), 18, 1, -1);
		multiply(resultC[0], te, resultC[0]);
		multiply(resultC[1], te, resultC[1]);
	}
	merge(resultC, 2, temp1);
	idft(temp1, temp1);
	split(temp1, resultC);
	magnitude(resultC[0], resultC[1], resultC[0]);
	normalize(resultC[0], resultC[0], 0, 1, NORM_MINMAX);
	//imshow("result", resultC[0]);
	output = resultC[0];
}
Mat max_min_functin(Mat img, int ksize, int flag)
{
	Mat img0, img1;
	int i = 0;
	if (flag == 0)i = 255;
	int x1;
	x1 = static_cast<int>(ksize / 2);
	img.copyTo(img0);
	copyMakeBorder(img0, img0, x1, x1, x1, x1, BORDER_CONSTANT, Scalar(i));
	img0.copyTo(img1);
	for (int i = x1; i < img0.rows - x1; ++i)
	{
		for (int j = x1; j < img0.cols - x1; ++j)
		{
			int temp = img0.at<uchar>(i, j);
			for (int k = i - x1; k <= i + x1; ++k)
			{
				for (int l = j - x1; l <= j + x1; ++l)
				{
					if (flag == 0)
					{
						if (img0.at<uchar>(k, l) < temp)
							temp = img0.at<uchar>(k, l);
					}
					else if (flag == 1)
					{
						if (img0.at<uchar>(k, l) > temp)
							temp = img0.at<uchar>(k, l);
					}
				}
			}
			img1.at<uchar>(i, j) = temp;
		}
	}
	Mat img2(img1, Rect(x1, x1, img.rows, img.cols));
	return img2;
}
Mat center_point(Mat img, int ksize)
{
	Mat temp1, temp2, temp3;
	temp1 = max_min_functin(img, ksize, 0);
	temp2 = max_min_functin(img, ksize, 1);
	temp3 = (temp1 + temp2) / 2;
	return temp3;
}
Mat pow_blur(Mat img, int ksize)
{
	Mat img0, img1;
	int x1 = static_cast<int>(ksize / 2), x = 255;
	img.copyTo(img0);
	copyMakeBorder(img0, img0, x1, x1, x1, x1, BORDER_CONSTANT, Scalar(x));
	img0.copyTo(img1);
	for (int i = x1; i < img0.rows - x1; ++i)
	{
		for (int j = x1; j < img0.cols - x1; ++j)
		{
			double temp = 1;
			for (int k = i - x1; k <= i + x1; ++k)
			{
				for (int l = j - x1; l <= j + x1; ++l)
				{
					temp *= img0.at<uchar>(k, l);
				}
			}
			auto pixel = pow(temp, 1.0 / (ksize * ksize));
			if (pixel < 0)pixel = 0;
			else if (pixel > 255) pixel = 255;
			img1.at<uchar>(i, j) = static_cast<uchar> (pixel);
		}
	}
	Mat img2(img1, Rect(x1, x1, img.rows, img.cols));
	return img2;
}
Mat cyclical_noise(Mat img)
{
	Mat temp = img.clone();
	int row = img.rows, col = img.cols;
	for (int i = 1; i < row; ++i)
	{
		for (int j = 1; j < col; ++j)
		{
			int pixel = img.at<uchar>(i, j) + 40 * sin(50 * i) /* + 40 * sin(50 * j)*/;
			if (pixel < 0)pixel = 0;
			else if (pixel > 255)pixel = 255;
			temp.at<uchar>(i, j) = pixel;
		}
	}
	return temp;
}
double getSNR(Mat& srcImage, Mat& dstImage)
{
	Mat src = dstImage;
	Mat dst = srcImage;
	int channels = dstImage.channels();
	int rowsNumber = src.rows;
	int colsNumber = src.cols * channels;

	double sigma = 0.0;
	double mse = 0.0;
	double SNR;
	for (int i = 0; i < rowsNumber; ++i)
	{
		for (int j = 0; j < colsNumber; ++j)
		{
			sigma += (src.ptr<uchar>(i)[j]) * (src.ptr<uchar>(i)[j]);
			mse += (src.ptr<uchar>(i)[j] - dst.ptr<uchar>(i)[j]) * (src.ptr<uchar>(i)[j] - dst.ptr<uchar>(i)[j]);
		}
	}
	SNR = 10 * log10(sigma / mse);
	return SNR;
}
void Shifting_DFT(Mat& amagnitude)
{
	int centerX = amagnitude.cols / 2;
	int centerY = amagnitude.rows / 2;
	Mat Qlt(amagnitude, Rect(0, 0, centerX, centerY));
	Mat Qrt(amagnitude, Rect(centerX, 0, centerX, centerY));
	Mat Qlb(amagnitude, Rect(0, centerY, centerX, centerY));
	Mat Qrb(amagnitude, Rect(centerX, centerY, centerX, centerY));
	Mat temp;
	Qlt.copyTo(temp);
	Qrb.copyTo(Qlt);
	temp.copyTo(Qrb);
	Qrt.copyTo(temp);
	Qlb.copyTo(Qrt);
	temp.copyTo(Qlb);
}
Mat YHomoFilter(Mat orginalImg)//YUV�ռ�
{
	Mat dst(orginalImg.rows, orginalImg.cols, CV_8UC3);
	cvtColor(orginalImg, orginalImg, COLOR_BGR2YUV);
	std::vector <Mat> yuvImg;
	split(orginalImg, yuvImg);
	yuvImg[0] = HomoFilter(yuvImg[0]);
	merge(yuvImg, dst);
	cvtColor(dst, dst, COLOR_YUV2BGR);
	return dst;
}
Mat HomoFilter(Mat inputImg) {
	Mat homo_result_Img, mat_dct, H_u_v;
	Mat homoImg = inputImg.clone();
	homoImg.convertTo(homoImg, CV_64FC1);
	int rows = homoImg.rows;
	int cols = homoImg.cols;
	int row = getOptimalDFTSize(homoImg.rows);
	int col = getOptimalDFTSize(homoImg.cols);
	copyMakeBorder(homoImg, homoImg, 0, row - rows, 0, col - cols, BORDER_CONSTANT, Scalar::all(0));
	homoImg += Scalar(1);
	log(homoImg, homoImg);
	// dct ��ɢ���ұ任
	dct(homoImg, mat_dct);
	// ��˹̬ͬ�˲���
	double gammaH = 3;//>1   ��Ƶ����
	double gammaL = 0.1;//<1   ��Ƶ����
	double C = 0.6; //б���񻯳��� б��
	double  d0 = (homoImg.rows / 2) * (homoImg.rows / 2) + (homoImg.cols / 2) * (homoImg.cols / 2);//��ֹƵ��
	//double  d0 = 150;//5-200  ��ֹƵ�� Խ��Խ��
	double  d2 = 0;
	H_u_v = Mat::zeros(rows, cols, CV_64FC1);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			d2 = pow((i - homoImg.rows / 2), 2.0) + pow((j - homoImg.cols / 2), 2.0);
			H_u_v.at<double>(i, j) = (gammaH - gammaL) * (1 - exp(-C * (d2 / d0))) + gammaL;
		}
	}
	H_u_v.at<double>(0, 0) = 1.1;
	mat_dct = mat_dct.mul(H_u_v);//��Ӧλ�˻�
	// idct
	idct(mat_dct, homo_result_Img);
	//exp
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			homo_result_Img.at<double>(i, j) = exp(homo_result_Img.at<double>(i, j));
		}
	}
	homo_result_Img.convertTo(homo_result_Img, CV_8UC1);
	return homo_result_Img;
}
Mat checker_generator(void)
{
	Mat checker = Mat::zeros(Size(512, 512), CV_8UC1);
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 512; ++j)
		{
			for (int k = 0; k < 64; ++k)
			{
				checker.at<uchar>(i * 64 + k, j) = (i % 2);
			}
		}
	}
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0; j < 512; ++j)
		{
			for (int k = 0; k < 64; ++k)
			{
				checker.at<uchar>(j, i * 64 + k) = ~(checker.at<uchar>(j, i * 64 + k)==i%2);
			}
		}
	}
	normalize(checker, checker, 255, 0, NORM_MINMAX);
	return checker;
}

void genaratePsf(Mat& psf, cv::Point& anchor, double len, double angle)
{
	//���ɾ���˺�ê��
	double half = len / 2;
	double alpha = (angle - floor(angle / 180) * 180) / 180 * CV_PI;
	double cosalpha = cos(alpha);
	double sinalpha = sin(alpha);
	int xsign;
	if (cosalpha < 0) {
		xsign = -1;
	}
	else {
		if (angle == 90) {
			xsign = 0;
		}
		else {
			xsign = 1;
		}
	}
	int psfwdt = 1;
	//ģ���˴�С
	int sx = (int)fabs(half * cosalpha + psfwdt * xsign - len * FLT_EPSILON);
	int sy = (int)fabs(half * sinalpha + psfwdt - len * FLT_EPSILON);
	cv::Mat_<double> psf1(sy, sx, CV_64F);

	//psf1�����Ͻǵ�Ȩֵ�ϴ�Խ�����½�ȨֵԽС�ĺˡ�
	//��ʱ�˶����Ǵ����½ǵ����Ͻ��ƶ�
	for (int i = 0; i < sy; i++) {
		double* pvalue = psf1.ptr<double>(i);
		for (int j = 0; j < sx; j++) {
			pvalue[j] = i * fabs(cosalpha) - j * sinalpha;

			double rad = sqrt(i * i + j * j);
			if (rad >= half && fabs(pvalue[j]) <= psfwdt) {
				double temp = half - fabs((j + pvalue[j] * sinalpha) / cosalpha);
				pvalue[j] = sqrt(pvalue[j] * pvalue[j] + temp * temp);
			}
			pvalue[j] = psfwdt + FLT_EPSILON - fabs(pvalue[j]);
			if (pvalue[j] < 0) {
				pvalue[j] = 0;
			}
		}
	}
	//    �˶��������������˶���ê���ڣ�0��0��
	anchor.x = 0;
	anchor.y = 0;
	//    �˶������������Ͻ��ƶ���ê��һ�������Ͻ�
	//    ͬʱ�����ҷ�ת�˺�����ʹ��Խ����ê�㣬ȨֵԽ��
	if (angle < 90 && angle>0) {
		flip(psf1, psf1, 1);
		anchor.x = psf1.cols - 1;
		anchor.y = 0;
	}
	else if (angle > -90 && angle < 0) {    //ͬ�������½��ƶ�
		flip(psf1, psf1, -1);
		anchor.x = psf1.cols - 1;
		anchor.y = psf1.rows - 1;
	}
	else if (angle < -90) {   //ͬ�������½��ƶ�
		flip(psf1, psf1, 0);
		anchor.x = 0;
		anchor.y = psf1.rows - 1;
	}
	/*����ͼ�����������䣬��һ������*/
	double sum = 0;
	for (int i = 0; i < sy; i++) {
		for (int j = 0; j < sx; j++) {
			sum += psf1[i][j];
		}
	}
	psf = psf1 / sum;
}

Mat WienerFilter(const Mat& src, const Mat& ref, int stddev)
{
	//��ЩͼƬ�ǹ����л��õ��ģ�pad��ԭͼ��0�����ͼ��cpx��˫ͨ��Ƶ��ͼ��mag��Ƶ���ֵͼ��dst���˲����ͼ��
	Mat pad, cpx, dst;

	//��ȡ����Ҷ�仯���ͼƬ�ߴ磬Ϊ2��ָ��
	int m = getOptimalDFTSize(src.rows);
	int n = getOptimalDFTSize(src.cols);

	//��ԭʼͼƬ��0�����������ѳߴ�ͼƬ
	copyMakeBorder(src, pad, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

	//��òο�ͼƬƵ��
	Mat tmpR(pad.rows, pad.cols, CV_8U);
	resize(ref, tmpR, tmpR.size());
	Mat refSpectrum = GetSpectrum(tmpR);

	//�������Ƶ��
	Mat tmpN(pad.rows, pad.cols, CV_32F);
	randn(tmpN, Scalar::all(0), Scalar::all(stddev));
	Mat noiseSpectrum = GetSpectrum(tmpN);

	//��src���и���Ҷ�任
	Mat planes[] = { Mat_<float>(pad), Mat::zeros(pad.size(), CV_32F) };
	merge(planes, 2, cpx);
	dft(cpx, cpx);
	split(cpx, planes);

	//ά���˲�����
	Mat factor = refSpectrum / (refSpectrum + noiseSpectrum);
	multiply(planes[0], factor, planes[0]);
	multiply(planes[1], factor, planes[1]);

	//���ºϲ�ʵ��planes[0]���鲿planes[1]
	merge(planes, 2, cpx);

	//���з�����Ҷ�任
	idft(cpx, dst, DFT_SCALE | DFT_REAL_OUTPUT);

	dst.convertTo(dst, CV_8UC1);
	return dst;
}
Mat GetSpectrum(const Mat& src)
{
	Mat dst, cpx;
	Mat planes[] = { Mat_<float>(src), Mat::zeros(src.size(), CV_32F) };
	merge(planes, 2, cpx);
	dft(cpx, cpx);
	split(cpx, planes);
	magnitude(planes[0], planes[1], dst);
	//Ƶ�׾���Ƶ�����ͼ��ƽ��
	multiply(dst, dst, dst);
	return dst;
}