//در این قسمت کتابخانههای لازم برای برنامه را وارد میکنیم
#include <opencv2\opencv.hpp>
#include <opencv\cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

// کتابخانه های وارد شده مربوط به دو کلاس 
// cv ,std
//هستند و برای اینکه قبل از هر تابع نیازی به نوشتن این دو مورد نباشه اول برنامه تعریفی از اون بیان میشه
using namespace cv;
using namespace std;

// تعریف توابعی که در تابع اصلی استفاده میشوند باید قبل از تابع اصلی باشه منظور از تابع اصلی 
//int main()

//حد آستانه در تابع هریس 
int thresh = 150;
int max_thresh = 255;
int herris_cal(Mat image, Mat img_gray); //تعریف تابع هریس
int Opt_Flow();
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color);
int Image_Pyramid(Mat image);
int Gray_Histogram(Mat &image);
int convolve(Mat image);
int hist2(Mat image);


//تابع اصلی
int main() {

	// تصاویر به صورت ماتریس هستند در اینجا دو ماتریس برای تصویر اصلی و تصویر خاکستری مقدار دهی اولیه میکنیم
	// mat عبارت
	//برای ساختن ماتریس است
	// ماتریس تصویر اصلی image
	//img_gray ماتریس تصویر خاکستری
	Mat image, img_gray;

	// با استفاده از این دستور تصویر مورد نظر را وارد برنامه میکنیم
	// imread(آدرس تصویر, 1);
	image = imread("7.jpg", 1); // Read the file


	// در این قسمت چک می شود که آیا تصویر وارد شده یا نه! در صورتی که ماتریس تصویر خالی باشد خطا می دهد
	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	//نمایش تصویر اصلی
	imshow("Orginal Image", image);

	// این تابع برای تغییر رنگ تصویر اصلی به تصویر خاکستری استفاده می شود
	//cvtColor(تصویر اصلی,j ماتریس تصویر خاکستری, Cکد تبدیل رنگی به خاکستری);
	cvtColor(image, img_gray, COLOR_BGR2GRAY);


	//نامگذاری پنجره باز شده برای نمایش تصویر خاکستری
	namedWindow("Gray Image", WINDOW_AUTOSIZE);

	//نمایش تصویر خاکستری
	imshow("Gray Image", img_gray);
	// در اپن سی وی برای نمایش عکس از این دستور بعد از خط قبلی استفاده می شود این دستور یک مکث خیلی کوچک 
	//برای نمایش تصویر ایجاد می کند
	waitKey(10);

	//در این قسمت باید الگوریتم مورد نظرتون رو انتخاب کنید
	//برای گرفتن شماره الگوریتم مورد نظر کاربر استفاده می شود a متغیر 
	// int  به معنی متغیر عدد صحیح است
	int a=1;

	// این حلقه برای گرفتن شماره الگوریتم مورد نظر کاربر استفاده می شود
	// تا زمانی که کاربر عدد 9 را وارد نکند برنامه متوقف نمی شود
	while (a != 9) {
		// این دستور برای نمایش در خروجی به کار میرودcout
		//endl برای نوشتن ادامه متن به خط بعدی می برد
		cout << "Enter the number of your request algorithm : " << endl;
		cout << "1- optical flow" << endl;
		cout << "2- herris algorithm " << endl;
		cout << "3- Image Pyramids" << endl;
		cout << "4- Image Convolution " << endl;
		cout << "5- Gray Image Histogram " << endl;
		cout << " if you want to close program enter '9' :    ";

		//cin برای گرفتن کاراکتر یا عدد از کاربر استفاده میشود
		cin >> a;

		// در این شرط با بررسی عدد ورودی توسط کاربر به هر یک از توابع متناظر با عدد می رود
		if (a == 1)
			Opt_Flow();
		else if (a == 2)
			herris_cal(image, img_gray);
		else if (a == 3)
			Image_Pyramid(image);
		else if (a == 4)
			convolve(image);
		else if (a == 5)
			Gray_Histogram(img_gray);
		
		
			// این دستور برای پاک کردن صفحه است
		system("CLS");
	}

	return 0;
}


// تابع هیستوگرام تصویر خاکستری 
// ورودی این تابع تصویر خاکستری است Mat &img_gray
int Gray_Histogram(Mat &img_gray) {

	cout << "************************************" << endl;
	cout << " ** Calculae Histogram ** \n ";

	// ساختن سه ماتریس اولیه برای خروجی هیستوگرام تصویر
	Mat hist,hsv,dst;
	
	//از آنجایی که رنگ ها از 0 تا 255 مقدار دهی می شوند بیشترین مقدار ماتریس هیستوگرام را به طور ثابت 256 در نظر می گیریم
	int const hist_height = 256;

	//
	Mat3b hist_image = Mat3b::zeros(hist_height, 256);

	// در این برنامه از تابع موجود در اپن سی وی برای محاسبه هیستو گرام استفاده شده است
	// متغیرهای بیان شده در زیر تماما متغیرهای لازم برای این تابع هستند
	// این متغیر محدوده هیستوگرام را نشان می دهد
	// در این تابع اکثر ورودی ها به صورت آرایه هستند لذا لازم است که ورودی ها مانند زیر تبدیل به آرایه شوند
	float hranges[] = { 0, 256 };
	//لازم است که این متغیر در یک آرایه قرار بگیرد
	const float* ranges[] = { hranges };

	// قرار دادن در یک تابع آرایه 
	int histSize[] = { hist_height };

	// CV_8U تبدیل فرمت تصویر به فرمت  
	// تابع محاسبه هیستوگرام این فرمت را به عنوان ورودی می پذیرد
	img_gray.convertTo(dst, CV_8U);

	// نمایش تصویر به دست آمده
	imshow("image", dst);
	waitKey(10);

	// محاسبه هیستوگرام
	//calcHist(jتصویر ورودی, one, 0, Mat(), ماتریس تصویر خروجی, one, آرایه ماکزیمم مقدار هیستوگرام, rآرایه محدوده هیستوگرام, true, false);
	calcHist(&dst, 1, 0, Mat(), hist, 1, histSize, ranges, true, false);

	//مقدار دهی اولیه به متغیر ماکزیمم مقدار
	double max_val = 0;

	// بدست آوردن ماکزیمم مقدار در ماتریس هیستوگرام
	minMaxLoc(hist, 0, &max_val);


	// حلقه برای رسم هیستوگرام تصویر
	for (int b = 0; b < 256; b++) {

		float const binVal = hist.at<float>(b);
		int   const height = cvRound(binVal*hist_height / max_val);

		//نمایش هر مقدار هیستوگرام با یک خط    
		//برای رسم هر خط دو نقطه نیاز است که نقطه اول مقدار ابتدای هیستوگرام و نقطه دوم نهایی هیستوگرام را نمایش میدهد
		//line(تصویراصلی , pنقطه اول, pنقطه دوم , cرنگ خط)
		line(hist_image, cv::Point(b, hist_height - height), cv::Point(b, hist_height), cv::Scalar::all(255));
	}
	//نمایش هیتوگرام تصویر
	imshow(" Histogram for Gray Image ", hist_image);
	
	//رفتن به تابع روش دوم برای به دست آوردن هیستوگرام 
	hist2(img_gray);

	//  ESC این مقدار برای خروج از تابع هیستوگرام است در صورتی که کاربر کلید
	// را انتخاب کند از تابع خارج می شود
	int c = waitKey(500);


	/// Press 'ESC' to exit the program
	if ((char)c == 27)
	{
		//بستن پنجره های تصویر هیستوگرام
		destroyWindow(" Histogram for Gray Image ");
		destroyWindow("Intensity Histogram");
		return 0;
	}

	// با کلیک بر تصویر از تابع خارج می شود
	waitKey();
	destroyWindow(" Histogram for Gray Image ");
	destroyWindow("Intensity Histogram");
	cout << "************************************" << endl;
	return 0;
}


// روش دوم(روش دستی) برای بدست آوردن هیستوگرام 
int hist2(Mat image) {
	// ساختن یک آرایه برای قرار دادن مقدار هیستوگرام تصویر
	int histogram[256];
	
	// مقدار دهی اولیه به آرایه
	// این حلقه مقدار هر یک از سلول های آرایه را صفر قرار می دهد
	for (int i = 0; i < 255; i++)
	{
		histogram[i] = 0;
	}


	// محاسبه شدت نور هر پیکسل : در این حلقه همه پیکسل های تصویر بررسی شده و تعداد پیکسل ها با شدت نوری مشابه
	// را ذخیره می کند به طوری مثال اگر در کل تصویر 2 پیکسل با شدت نور 200 وجود داشته باشد 
	// مقدار هیستوگرام آن2 است
	for (int y = 0; y < image.rows; y++)
		for (int x = 0; x < image.cols; x++)
			histogram[(int)image.at<uchar>(y, x)]++;

	// نمایش هیستوگرام هر پیکسل در خروجی
	for (int i = 0; i < 256; i++)
		cout << histogram[i] << " ";

	// کشیدن نمودار هیستوگرام

	//مقدار دهی اولیه برای طول و عرض ماتریس هیستوگرام
	int hist_w = 512; int hist_h = 400;

	// ساخت ماتریس تصویر هیستوگرام 
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));

	//cvRound برای گرد کردن اعداد استفاده می شود
	int bin_w = cvRound((double)hist_w / 256);
	

	// پیدا کردن ماکزیمم مقدار در نمودار هیستوگرام 
	int max = histogram[0];
	for (int i = 1; i < 256; i++) {
		if (max < histogram[i]) {
			max = histogram[i];
		}
	}

	// نرمال کردن مقدار هیستوگرام 

	for (int i = 0; i < 255; i++) {
		histogram[i] = ((double)histogram[i] / max)*histImage.rows;
	}


	// رسم خط های شدت نور در نمودار هیستوگرام
	for (int i = 0; i < 255; i++)
	{
		line(histImage, Point(bin_w*(i), hist_h),
			Point(bin_w*(i), hist_h - histogram[i]),
			Scalar(0, 0, 0), 1, 8, 0);
	}

	// رسم نمودار هیستوگرام
	namedWindow("Intensity Histogram", CV_WINDOW_AUTOSIZE);
	imshow("Intensity Histogram", histImage);

	waitKey(1);
	return 0;

}


//Mat image تابع کانولوشن تصویر : ورودی تصویر اصلی 
// در این تابع یک تصویر در یک فیلتر ضرب خواهد شد که به این عمل کانولوشن تصویر میگویند

int convolve(Mat image) {
	cout << "************************************" << endl;
	cout << " ** Convolution Image ** \n ";

	// ساختن ماتریس جدید برای تصویر کانولوشن شده
	Mat dst;

	/// مقدار دهی اولیه به مقادیر فیلتر اعمال شده به تصویر

	//نقطه ثابت فیلتر که بر اساس آن فیلتر جا به جا می شود و معمولا نقطه وسط ماتریس فیلتر است و به صورت پیشفرض 
	//(-1, -1) است
	//Point دستور نقطه و یا موقعیت ایجاد می کند
	Point anchor = Point(-1, -1);

	// مقداری که به هر پیکسل تصویر در حین عمل کانولوشن اضافه می شود و به صورت پیش فرض صفر است
	double delta = 0;

	// مقدار دهی اولیه به عمق تصویر- هر تصویر رنگی از سه کانال رنگی تشکیل شده پس عمق یک تصویر معمولی 3 است
	//در صورتی که بخواهید عمق تصویر همان عمق تصویر ورودی باشد این مقدار را -1 قرار می دهیم
	int ddepth = -1;

	// اندازه فیلتر
	int kernel_size;
	
	//شمارنده برای بدست آوردن ماتریس کرنل جدید
	int ind = 0 ,c;

	//حلقه بی نهایت تا زمانی که کاربر کلید ESC را فشار دهد
	while (true)
	{

		c = waitKey(500);
		/// Press 'ESC' to exit the program
		if ((char)c == 27)
		{
			destroyWindow("Image convolution ");
			break;
		}


		/// در هر مرحله سایز ماتریس فیلتر را تغییر می دهد برای اینکه به تصاویر مختلف برسیم
		kernel_size = 3 + 2 * (ind % 5);

		// ساخت ماتریس کرنل یا فیلتر : این روش یکی از روش های ساختن ماتریس رندوم بر اساس سایز ماتریس کرنل است 
		// به این صورت که اول یک ماتریس همانی (همه ماتریس 1) میسازد و سپس بر مقدار توان دوم اندازه ماتریس تقسیم میکند
		Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);

		/// اعمال فیلتر بر تصویر : این دستور این کار را انجام می دهد
		// filter2D(تصویر اصلی, dتصویر خروجی کانولوشن, dعمق تصویر خروجی, kماتریس فیلتر, anchor, delta, BORDER_DEFAULT);
		filter2D(image, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT);

		//نمایش تصویر کانولوشن شده
		imshow("Image convolution ", dst);

		//اضافه کردن شمارنده برای به دست آوردن ماتریس کرنل جدید
		ind++;
	}


	cout << "************************************" << endl;
	destroyWindow("Image convolution ");
	return 0;
}


//تابع پیرامید تصویر 

int Image_Pyramid(Mat src) {

	//i راهنمای تابع : در صورتی که میخواهید تصویر را بزرگ یا کوچک کنید ابتدا روی تصویر کلیک کرده و سپس کلید 
	//o را برای بزگتر شدن تصویر و کلید 
	//را برای کوچک تر شدن تصویر بزنید

	cout << "************************************" << endl;
	cout << "\n Zoom In-Out demo ( Image Pyramid) \n "
		"------------------  \n"
		" * for Zoom in enter  'i'   \n"
		" * for Zoom out  enter  'o'  \n"
		" * [ESC] -> Close program \n" << endl;

	//نمایش تصویر پیرامید 
	imshow("Image Pyramid ", src);
	waitKey(1);


	for (;;)
	{
		//ESC کاربر در هر لحظه با زدن کلید 
		// می تواند از برنامه خارج شود
		char c = (char)waitKey(0);
		if (c == 27)
		{
			//بستن پنجره تصویر پیرامید
			destroyWindow("Image Pyramid ");
			break;
		}
		//در صورتی که i را فشار داد
		else if (c == 'i')
		{
			//دو برابر کردن تصویر با استفاده از تابع پیرامید
			//pyrUp(تصویر ورودی, sتصویر خروجی, Sسایز تصویر خروجی که دو برابر تصویر ورودی است);
			pyrUp(src, src, Size(src.cols * 2, src.rows * 2));
			cout << "** Zoom In: Image x 2 \n";
		}
		// o در صورتی که کلید را فشار داد    
		else if (c == 'o')
		{
			// تابع پیرامید برای کوچک کردن تصویر
			//pyrDown(تصویر ورودی, sتصویر خروجی, Sسایز تصویر خروجی که نصف تصویر ورودی است);
			pyrDown(src, src, Size(src.cols / 2, src.rows / 2));
			cout << "** Zoom Out: Image / 2 \n";
		}
		//نمایش تصویر پیرامید
		imshow("Image Pyramid ", src);
		waitKey(1);
	}
	
	cout << "************************************" << endl;
	//بستن پنجره تصویر پیرامید
	destroyWindow("Image Pyramid ");
	return -1;
}


// Optical Flow تابع رسم 
//Opt_Flow() ورودی های این تابع از تابع اصلی 
// هستند و شامل تصویر جریان نوری و تصویر فریم قبلی و رنگ مورد نظر برای رسم فلش ها در تصویر است
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color)
{
	// جریان نوری flow
	//cflowmap تصویر اصلی از فریم قبل
	//بررسی تمام پیکسل های تصویر 
	for (int y = 0; y < cflowmap.rows; y += step)
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			// را بدست آورده  x,y مقدار جریان نوری در پیکسل 
			const Point2f& fxy = flow.at<Point2f>(y, x);

			//رسم خط و دایره نشان دهنده جریان نوری و حرکت در تصویر
			//برای رسم هر خط دو نقطه نیاز است که نقطه اول موقعیت شی در موقعیت فریم قبل و نقطه دوم مقدار حرکت هر فریم را نمایش میدهد
			//line(تصویراصلی , pنقطه اول, pنقطه دوم , cرنگ خط)

			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),color);

			//رسم نقطه برای نشان دادن جریان نوری
			circle(cflowmap, Point(x, y), 2, color, -1);
		}
}


// Optical Flow تابع اصلی 
//نکته ای که لازم است ذکر شود جریان نوری میزان تغییر پیکسل در هر فریم را بررسی میکند

int Opt_Flow() {
	cout << "************************************" << endl;
	cout << " ** Optical Flow ** \n ";

	//VideoCapture در این قسمت کلاس 
	// از کتابخانه اپن سی وی را نیاز داریم 
	//cap نام متغیری است که ویدیوی ورودی را در آن قرار می دهیم
	//در داخل پرانتز آدرس ویدیوی مورد نظر را قرار می دهیم 
	VideoCapture cap("vtest.avi");

	//در این قسمت چک میکنیم که آیا ویدیو باز شده یا نه 
	// در صورت خالی بودن متغیر خطا میدهد
	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	// در اینجا سه ماتریس ایجاد میشود
	//frame ماتریس برای گرفتن فریم های تصویر است
	// flow, cflow و ماتریس های روبرو برای تشخیص حرکت و تغییر پیکسل ها استفاده می شود 
	Mat flow, cflow, frame;

	//ماتریس هایی که برای تبدیل تصویر نیاز است
	//gray ماتریس تبدیل تصویر به حاکستری
	//prevgray ماتریس تصویر خاکستری فریم قبل
	UMat gray, prevgray, uflow;

	//باز کردن پنجره برای نمایش فیلم
	namedWindow("flow", 1);

	//حلقه تا زمانی که فیلم به پایان برسد
	for (;;)
	{
		//گرفتن هر فریم در فیلم
		cap >> frame;

		//اگر فریم خالی بود یا فیلم تمام شده بود از حلقه خارج می شود
		if (frame.empty())
			break;

		//تبدیل رنگ تصویر فریم به خاکستری
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		//اگر ماتریس خاکستری فریم قبل خالی نیست
		if (!prevgray.empty())
		{
			//Gunnar Farneback محاسبه جریان نوری فیلم با استفاده از الگوریتم 
			//calcOpticalFlowFarneback(تصویر ورودی فریم قبلی, gتصویر ورودی خاکستری فریم , uتصویر خروجی که هم سایز تصویر خاکستری جدید است , l0.5, 3, 15, 3, 5, 1.2, 0);
			calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);

			//تبدیل دوباره تصویر خاکستری فریم قبل به تصویر رنگی
			cvtColor(prevgray, cflow, COLOR_GRAY2BGR);

			//کپی کردن ماتریس جریان نوری در یک ماتریس جدید  
			uflow.copyTo(flow);

			//drawOptFlowMap رفتن به تابع برای رسم جریان نوری
			drawOptFlowMap(flow, cflow, 16, Scalar(0, 255, 0));
			
			//نمایش فریم با شار نوری
			imshow("flow", cflow);
			
		}


		int c = waitKey(1);
		/// Press 'ESC' to exit the program
		if ((char)c == 27)
		{
			
			
			destroyWindow("flow");
			return 0;
		}
		
		//کپی کردن فریم جدید در ماتریس فریم قبلی
		swap(prevgray, gray);
	}
	cout << "************************************" << endl;
	//بستن پنجره
	destroyWindow("flow");
	return -1;
}


//محاسبه تابع هریس : ورودی ها تصویر اصلی و تصویر خاکستری
int herris_cal(Mat image, Mat img_gray) {

	cout << "************************************" << endl;
	cout << " ** herris algorithm ** \n ";

	/// مقدار دهی اولیه مقادیر تابع

	//همسایگی در نظر گرفته شده
	int blockSize = 2;

	//مقدار سایز فیلتر اعمالی
	int apertureSize = 3;

	//پارامتر مربوط به تابع به صورت پیش فرض همین مقدار است
	double k = 0.04;

	/// تشحیص گوشه 
	//ساخت یک ماتریس خروجی به همان ابعاد تصویر اصلی با مقدار اولیه صفر 
	Mat dst = Mat::zeros(image.size(), CV_32FC1);

	// تابع به دست آوردن گوشه 
	//cornerHarris(تصویر خاکستری ورودی, dماتریس تصویر خروجی, blockSize, apertureSize, k);
	cornerHarris(img_gray, dst, blockSize, apertureSize, k);

	/// نرمال کردن ماتریس
	// ساخت دو ماتریس برای نرمال کردن
	Mat dst_norm, dst_norm_scaled;
	
	//دستور نرمال کردن تصویر به دست آمده از تابع هریس
	//normalize(تصویر ورودی, dماتریس خروجی نرمال شده, k0, رنج رنگی, NORM_MINMAX, CV_32FC1, Mat());
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	//هم رنج کردن مقادیر و قرار دادن همه داده ها در یک بازه
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// کشیدن دایره دور گوشه های پیدا شده در تصویر
	//حلقه به تعداد گوشه های پیدا شده در تصویر ادامه می یابد
	for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			//اگر مقدار بدست آمده برای پیکسل از تابع هریس از مقدار آستانه بیشتر باشد آن را یک گوشه در نظر می گیرد
			if ((int)dst_norm.at<float>(i, j) > thresh)
			{
				//رسم دایره 
				//circle(تصویر وردی, Pنقطه مورد نظر برای رسم دایره : مرکز دایره, rشعاع دایره, Scalar(0), 2, 8, 0);
				circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
			}
		}
	}

	/// نمایش نتایج 
	namedWindow("Corners detected image with herris algorithm");
	imshow("Corners detected image with herris algorithm", dst_norm_scaled);
	cout << "************************************" << endl;

	int c = waitKey(50);
	/// Press 'ESC' to exit the program
	if ((char)c == 27)
	{
		destroyWindow("Corners detected image with herris algorithm");
		return 0;
	}
	waitKey();
	destroyWindow("Corners detected image with herris algorithm");
	return -1;
}

