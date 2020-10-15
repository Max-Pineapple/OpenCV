#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp" 
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>  //文件流，文件输入与输出

using namespace cv;
using namespace std;

//用于检测深度的图像以及进行极线校正的两张图像
const char* imageName_L = "left02.jpg";
const char* imageName_R = "right02.jpg";

const char* imageList_L = "caliberationpics_L.txt"; // 左相机的标定图片名称列表
const char* imageList_R = "caliberationpics_R.txt"; // 右相机的标定图片名称列表

const char* singleCalibrate_result_L = "calibrationresults_L.txt"; // 存放左相机的标定结果
const char* singleCalibrate_result_R = "calibrationresults_R.txt"; // 存放右相机的标定结果

const char* stereoCalibrate_result_L = "stereocalibrateresult.txt"; // 存放立体标定结果（双目）
//const char* stereoCalibrate_result_R = "stereocalibrateresult_R.txt";

const char* stereoRectifyParams = "stereoRectifyParams.txt"; // 存放立体校正参数，旋转与平移

vector<vector<Point2f>> corners_seq_L; // 建立一个存放角点的容器，存放左相机中所有角点坐标
vector<vector<Point2f>> corners_seq_R;
vector<vector<Point3f>> objectPoints_L;  // 左相机拍摄物体恢复的三维坐标
vector<vector<Point3f>> objectPoints_R;

/**************************************01 参数解析*******************************************************************/
/*
1、Scalar::all(0)就是给每个通道都赋值0
2、CV_32FC1表示图像类型是32位浮点型单通道的灰度图像
通道表示每个点能存放多少个数，类似于RGB彩色图中的每个像素点有三个值，即三通道的，那么单通道是指每个点只能存放1个数；
图片中的深度表示每个值由多少位来存储，是一个精度问题，一般图片是8bit（位）的，则深度是8，这里是32位，则深度是32
*/

Mat cameraMatrix_L = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 相机的内参数，此时的3*3内参数矩阵是包括焦距f、fx、fy、以及主点坐标cx、cy
Mat cameraMatrix_R = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 初始化相机的内参数
Mat distCoeffs_L = Mat(1, 5, CV_32FC1, Scalar::all(0)); // 左相机的畸变系数
Mat distCoeffs_R = Mat(1, 5, CV_32FC1, Scalar::all(0)); // 初始化右相机的畸变系数

Mat R, T, E, F; // 立体标定参数
Mat R1, R2, P1, P2, Q; // 立体校正参数，极线校正BOUGET


Mat mapl1, mapl2, mapr1, mapr2; // 图像重投影映射表
Mat img1_rectified, img2_rectified, disparity, result3DImage,result3DImage1; // 校正图像 视差图 深度图

Size patternSize = Size(6, 9); // 行列内角点个数，针对标定的棋盘格的行列
Size chessboardSize = Size(30, 30); // 棋盘上每个棋盘格的大小30mm（初始值，固定）

Size imageSize; // 图像尺寸（分辨率）

Rect validRoi[2];//图像校正之后,会对图像进行裁剪,这里的validROI就是指裁剪之后的区域

/*
单目标定： singleCameraCalibrate
参数：
	imageList		存放标定图片名称的txt，包括左标定和右标定
	singleCalibrateResult	存放标定结果的txt
	objectPoints	世界坐标系中点的坐标（已知）
	corners_seq		存放图像中的角点,用于立体标定
	cameraMatrix	相机的内参数矩阵，3*3
	distCoeffs		相机的畸变系数1*5
	imageSize		输入图像的尺寸（像素）
	patternSize		标定板每行的角点个数, 标定板每列的角点个数 (9, 6)
	chessboardSize	棋盘上每个方格的边长（mm），30mm

函数解析：
	1、getline函数：getline(cin, inputLine)，其中cin是正在读取的输入流，而inputLine是接收输入字符串的string变量的名称。

注意：亚像素精确化时，允许输入单通道，8位或者浮点型图像。
由于输入图像的类型不同，下面用作标定函数参数的内参数矩阵和畸变系数矩阵在初始化时也要注意数据类型。
*/
bool singleCameraCalibrate(const char* imageList, const char* singleCalibrateResult, vector<vector<Point3f>>& objectPoints,
	vector<vector<Point2f>>& corners_seq, Mat& cameraMatrix, Mat& distCoeffs, Size& imageSize, Size patternSize, Size chessboardSize)
{

	int n_boards = 0;

	//文件操作
	ifstream imageStore(imageList); // 打开存放标定图片名称的txt
	ofstream resultStore(singleCalibrateResult); // 保存标定结果的txt

	/*********************************************01 提取标定板上的角点坐标***********************************************/

	vector<Point2f> corners; // 存放一张图片的角点坐标 
	string imageName; // 读取的标定图片的名称，主要是针对imageList文件中的文件名称的读取

	while (getline(imageStore, imageName)) // 读取txt的每一行（每一行存放了一张标定图片的名称）
	{
		//记录读取的图片数量
		n_boards++;

		//创建临时变量，存放读取到的每一行标定图片
		Mat imageInput = imread(imageName);
		cvtColor(imageInput, imageInput, CV_RGB2GRAY); //转换成灰度图像，依然用 imageInput保存

		//获取图片尺寸，形参imageSize已经完成任务
		imageSize.width = imageInput.cols; // 获取图片像素的宽度
		imageSize.height = imageInput.rows; // 获取图片像素的高度

		// 检查棋盘格标定板的角点是否被找到，patterSize全局定义（9*6），将检查到的棋盘格角点存放于corners中
		bool found = findChessboardCorners(imageInput, patternSize, corners);
		// findChessboardCorners（）函数中的最后一个参数int flags的缺省值（默认值）为：CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE

		// 亚像素精确化，在findChessboardCorners中自动调用了cornerSubPix，为了更加精细化，我们自己再调用一次；
		if (found) // 当所有的角点（9*6）都被找到
		{
			TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001); // 终止标准，迭代40次或者达到0.001的像素精度
			cornerSubPix(imageInput, corners, Size(11, 11), Size(-1, -1), criteria);
			// 计算亚像素角点时考虑的区域的大小,域大小为 N*N; N=(winSize*2+1)，由于我们的图像尺寸较大，将搜索窗口调大一些，（11， 11）为真实窗口的一半，真实大小为（11*2+1， 11*2+1）--（23， 23）

			corners_seq.push_back(corners); // 存入角点序列，入栈操作，即将获取的corners按照入栈（先进后出）的方式存入corners_seq新的容器中

			// 绘制棋盘格的角点
			drawChessboardCorners(imageInput, patternSize, corners, true);
			imshow("cornersFrame", imageInput);
			waitKey(500); // 暂停0.5s，可以每张图测量角点之间的具体形状
		}

	}

	for (size_t i = 0; i < corners.size(); i++)
	{
		printf("第%d个点（%f,%f)\n", i, corners[i].x, corners[i].y);
	}
	//destroyWindow("cornersframe");

	/***********************************02 开始进行单目相机标定***********************************************/
	//1、 计算角点对应的三维坐标

	int pic;
	size_t i, j;
	for (pic = 0; pic < n_boards; pic++)  //n_boards:此时代表标定图片的数量
	{
		//创建存放标定图片上角点三维坐标：realPointSet
		vector<Point3f> realPointSet;

		//	patternSize :  标定板每行的角点个数, 标定板每列的角点个数 (9, 6)，实际为（6*9）
		for (i = 0; i < patternSize.height; i++) //6
		{
			for (j = 0; j < patternSize.width; j++) //9
			{
				Point3f realPoint;

				// 假设标定板位于世界坐标系Z=0的平面，此考虑平面x、y的坐标，并且以棋盘格左上角为初始坐标原点
				realPoint.x = j * chessboardSize.width;
				realPoint.y = i * chessboardSize.height;
				realPoint.z = 0;
				realPointSet.push_back(realPoint); // 存入角点的三维坐标，入栈操作，即将获取的realPoint按照入栈（先进后出）的方式存入realPointSet中
			}
		}

		objectPoints.push_back(realPointSet); //将检测到的角点三维坐标存放到世界坐标系下新的容器中，也就是换个名称继续存储检测角点的三维坐标
	}

	// 执行单目标定程序
	vector<Mat> rvec; // 旋转向量，因为每个vector<Point3f>会得到一个rvecs
	vector<Mat> tvec; // 平移向量， 因为每个vector<Point3f>会得到一个tvecs

	//在使用calibrateCamera标定前，一般使用findChessboardCorners()函数获得棋盘标定板的角点位，上述已完成角点位置的检测
	calibrateCamera(objectPoints, corners_seq, imageSize, cameraMatrix, distCoeffs, rvec, tvec, 0);

	// 保存单目标定结果
	resultStore << "相机内参数矩阵" << endl;
	resultStore << cameraMatrix << endl << endl;
	resultStore << "相机畸变系数" << endl;
	resultStore << distCoeffs << endl << endl;

	// 计算重投影点，与原图角点比较，得到误差
	double errPerImage = 0.; // 每张图像的误差初始化
	double errAverage = 0.; // 所有图像的平均误差初始化
	double totalErr = 0.; // 误差总和

	vector<Point2f> projectImagePoints; // 重投影点，二维点

	//标定图片数量下
	for (i = 0; i < n_boards; i++)
	{
		vector<Point3f> tempObjectPoints = objectPoints[i]; // 临时三维点，相当于每幅图像中都存放着一组棋盘格的角点坐标

		// 计算重投影点，函数projectPoints()根据所给的3D坐标和已知的几何变换来求解投影后的2D坐标存放于projectImagePoints，这是新的投影点
		projectPoints(tempObjectPoints, rvec[i], tvec[i], cameraMatrix, distCoeffs, projectImagePoints);

		// 计算新的投影点与旧的投影点之间的误差
		vector<Point2f> tempCornersPoints = corners_seq[i];// 临时存放旧投影点，第i幅图像中的一组旧投影点

		// 定义成两个通道的Mat(重新定义两个别名存放对比投影点)是为了计算误差
		Mat tempCornersPointsMat = Mat(1, tempCornersPoints.size(), CV_32FC2);
		Mat projectImagePointsMat = Mat(1, projectImagePoints.size(), CV_32FC2);

		// 新旧投影点进行双向赋值
		for (int j = 0; j < tempCornersPoints.size(); j++)
		{
			projectImagePointsMat.at<Vec2f>(0, j) = Vec2f(projectImagePoints[j].x, projectImagePoints[j].y);
			tempCornersPointsMat.at<Vec2f>(0, j) = Vec2f(tempCornersPoints[j].x, tempCornersPoints[j].y);
		}

		// opencv里的norm函数（范数函数求解）其实把这里的两个通道分别分开来计算(X1-X2)^2的值，然后统一求和，最后进行根号
		errPerImage = norm(tempCornersPointsMat, projectImagePointsMat, NORM_L2) / (patternSize.width * patternSize.height);
		totalErr += errPerImage;
		resultStore << "第" << i + 1 << "张图像的平均误差为：" << errPerImage << endl;
	}
	resultStore << "全局平局误差为：" << totalErr / n_boards << endl;
	imageStore.close();
	resultStore.close();
	return true;
}

/*
双目标定:计算两摄像机相对旋转矩阵 R,平移向量 T, 本征矩阵E, 基础矩阵F
参数：
	stereoCalibrateResult	存放立体标定结果的txt（待求）
	objectPoints			三维点（单目标定已经获取）
	imagePoints				二维图像上的点（单目标定已经获取）
	cameraMatrix			相机内参数（单目标定已经获取）
	distCoeffs				相机畸变系数（单目标定已经获取）
	imageSize				图像尺寸（已知量）

	R		左右相机相对的旋转矩阵
	T		左右相机相对的平移向量

	理论上：极限校正需要用到本质矩阵与基础矩阵
	E		本征矩阵（本质矩阵），本质矩阵（本征矩阵）是在归一化图像坐标下的基本矩阵，不仅具有基本矩阵的所有性质，并且可以估计两个相机的相对位置关系。
	F		基础矩阵
*/
bool stereoCalibrate(const char* stereoCalibrateResult, vector<vector<Point3f>> objectPoints, vector<vector<Point2f>> imagePoints1, vector<vector<Point2f>> imagePoints2,
	Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2, Size& imageSize, Mat& R, Mat& T, Mat& E, Mat& F)
{
	//文件操作，将双目标定的结果存于stereoCalibrateResult
	ofstream stereoStore(stereoCalibrateResult);

	//停止优化准则，迭代算法的终止条件
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 1e-6); // 终止条件是迭代30次，像素精度优化至0.000001

	//CALIB_FIX_INTRINSIC 表示固定输入的cameraMatrix和distCoeffs不变，只求解R,T,E,F
	stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1,
		cameraMatrix2, distCoeffs2, imageSize, R, T, E, F, CALIB_FIX_INTRINSIC, criteria); // 注意参数顺序，可以到保存的文件中查看，避免返回时出错
	stereoStore << "左相机内参数：" << endl;
	stereoStore << cameraMatrix1 << endl;
	stereoStore << "右相机内参数：" << endl;
	stereoStore << cameraMatrix2 << endl;
	stereoStore << "左相机畸变系数：" << endl;
	stereoStore << distCoeffs1 << endl;
	stereoStore << "右相机畸变系数：" << endl;
	stereoStore << distCoeffs2 << endl;
	stereoStore << "旋转矩阵：" << endl;
	stereoStore << R << endl;
	stereoStore << "平移向量：" << endl;
	stereoStore << T << endl;
	stereoStore << "本质矩阵：" << endl;
	stereoStore << E << endl;
	stereoStore << "基础矩阵：" << endl;
	stereoStore << F << endl;
	stereoStore.close();
	return true;
}

/*
立体校正：极线校正
参数：
	stereoRectifyParams	存放立体校正结果的txt
	cameraMatrix			相机内参数（双目获知）
	distCoeffs				相机畸变系数（双目获知）
	imageSize				图像尺寸（双目获知）
	R						左右相机相对的旋转矩阵（双目获知）
	T						左右相机相对的平移向量（双目获知）

	R1, R2					行对齐旋转校正（待求）
	P1, P2					左右投影矩阵（待求）
	Q						重投影矩阵（待求）
	map1, map2		左右图像的重投影映射表（待求）
*/
Rect stereoRectification(const char* stereoRectifyParams, Mat& cameraMatrix1, Mat& distCoeffs1, Mat& cameraMatrix2, Mat& distCoeffs2,
	Size& imageSize, Mat& R, Mat& T, Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2)
{

	Rect validRoi[2];//图像校正之后,会对图像进行裁剪,这里的validROI就是指裁剪之后的区域，2是指左右两幅图像都需要进行立体矫正

	//文件操作，将校正结果存入stereoRectifyParams文本中
	ofstream stereoStore(stereoRectifyParams);


	//当畸变系数和内外参数矩阵标定完成后，就应该进行畸变的矫正，以达到消除畸变的目的
	stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize,
		R, T, R1, R2, P1, P2, Q, 0, -1, imageSize, &validRoi[0], &validRoi[1]);


	// 计算左右图像的重投影映射表
	stereoStore << "R1：" << endl;
	stereoStore << R1 << endl;
	stereoStore << "R2：" << endl;
	stereoStore << R2 << endl;
	stereoStore << "P1：" << endl;
	stereoStore << P1 << endl;
	stereoStore << "P2：" << endl;
	stereoStore << P2 << endl;
	stereoStore << "Q：" << endl;
	stereoStore << Q << endl;

	//文件关闭操作
	stereoStore.close();

	cout << "R1:" << endl;
	cout << R1 << endl;
	cout << "R2:" << endl;
	cout << R2 << endl;
	cout << "P1:" << endl;
	cout << P1 << endl;
	cout << "P2:" << endl;
	cout << P2 << endl;
	cout << "Q:" << endl;
	cout << Q << endl;

	//求解出 mapl1, mapl2，mapr1, mapr2
	initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, CV_32FC1, mapl1, mapl2);
	initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, CV_32FC1, mapr1, mapr2);

	return validRoi[0], validRoi[1];//返回图像校正之后,会对图像进行裁剪,这里的validROI就是指裁剪之后的区域
}

/*
计算视差图
参数：
	imageName1	左相机拍摄的图像
	imageName2	右相机拍摄的图像
	img1_rectified	重映射后的左侧相机图像
	img2_rectified	重映射后的右侧相机图像
	map	重投影映射表
*/

bool computeDisparityImage(const char* imageName1, const char* imageName2, Mat& img1_rectified,
	Mat& img2_rectified, Mat& mapl1, Mat& mapl2, Mat& mapr1, Mat& mapr2, Rect validRoi[2], Mat& disparity)
{
	// 首先，对左右相机的两张图片进行重构
	Mat img1 = imread(imageName1);
	Mat img2 = imread(imageName2);

	//判断图像是否为空
	if (img1.empty() | img2.empty())
	{
		cout << "图像为空" << endl;
	}

	//将原图像转化成灰度图像（彩色图像转灰色图像）
	Mat gray_img1, gray_img2;
	cvtColor(img1, gray_img1, COLOR_BGR2GRAY);
	cvtColor(img2, gray_img2, COLOR_BGR2GRAY);


	// canvas：画布
	Mat canvas(imageSize.height, imageSize.width * 2, CV_8UC1); // 注意数据类型，8位无符号单通道数据类型
	Mat canLeft = canvas(Rect(0, 0, imageSize.width, imageSize.height));
	Mat canRight = canvas(Rect(imageSize.width, 0, imageSize.width, imageSize.height));

	//复制，将这两幅灰度图像嵌入到所建立的画布中
	gray_img1.copyTo(canLeft);
	gray_img2.copyTo(canRight);

	//imwrite函数是用来输出图像到文件，即保存创建的canvas图片到该工程文件夹下
	imwrite("校正前左右相机图像.jpg", canvas);

	//重映射，将一幅图像内的像素点放置到另外一幅图像内的指定位置即为重映射，线性插值
	remap(gray_img1, img1_rectified, mapl1, mapl2, INTER_LINEAR);
	remap(gray_img2, img2_rectified, mapr1, mapr2, INTER_LINEAR);

	imwrite("左相机校正图像.jpg", img1_rectified);
	imwrite("右相机校正图像.jpg", img2_rectified);

	img1_rectified.copyTo(canLeft);
	img2_rectified.copyTo(canRight);

	//rectangle函数是用来绘制一个矩形框的，通常用在图片的标记上，Scalar(255, 255, 255)白色，矩形线宽为5
	rectangle(canLeft, validRoi[0], Scalar(255, 255, 255), 5, 8);
	rectangle(canRight, validRoi[1], Scalar(255, 255, 255), 5, 8);

	//canvas.rows:指高度，一列480个像素，以16倍增，极线校正30条
	for (int j = 0; j <= canvas.rows; j += 16)
		line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(255, 0, 0), 1, 8); //线的颜色：蓝色
	imwrite("校正后左右相机图像.jpg", canvas);

	// 进行立体匹配，BM匹配算法
	Ptr<StereoBM> bm = StereoBM::create(16, 9);  // Ptr<>是一个智能指针模板，指定StereoBM算法
	bm->compute(img1_rectified, img2_rectified, disparity); // 计算视差图

	//数据类型的转换：CV_32F
	disparity.convertTo(disparity, CV_32F, 1.0 / 16);

	// 归一化视差映射
	normalize(disparity, disparity, 0, 256, NORM_MINMAX, -1);
	return true;

}

// 鼠标回调函数，点击视差图显示深度
void onMouse(int event, int x, int y, int flags, void* param)
{
	Point point;
	point.x = x;
	point.y = y;
	if (event == EVENT_LBUTTONDOWN)
	{
		//访问元素深度值
		cout << result3DImage.at<Vec3f>(point) << endl;
	}
}


int main()
{
	singleCameraCalibrate(imageList_L, singleCalibrate_result_L, objectPoints_L, corners_seq_L, cameraMatrix_L,
		distCoeffs_L, imageSize, patternSize, chessboardSize);
	cout << "已完成左相机的标定!" << endl;
	singleCameraCalibrate(imageList_R, singleCalibrate_result_R, objectPoints_R, corners_seq_R, cameraMatrix_R,
		distCoeffs_R, imageSize, patternSize, chessboardSize);
	cout << "已完成右相机的标定!" << endl;


	stereoCalibrate(stereoCalibrate_result_L, objectPoints_L, corners_seq_L, corners_seq_R, cameraMatrix_L, distCoeffs_L,
		cameraMatrix_R, distCoeffs_R, imageSize, R, T, E, F);
	cout << "相机立体标定完成！" << endl;//右相机相对左相机做的立体标定，获取R, T, E, F


	//stereoCalibrate(stereoCalibrate_result_R, objectPoints_R, corners_seq_L, corners_seq_R, cameraMatrix_L, distCoeffs_L,
	//	cameraMatrix_R, distCoeffs_R, imageSize, R2, T2, E2, F2);
	//cout << "右相机立体标定完成！" << endl;

	//返回值本身（就地操作）
	validRoi[0], validRoi[1] = stereoRectification(stereoRectifyParams, cameraMatrix_L, distCoeffs_L, cameraMatrix_R, distCoeffs_R,
		imageSize, R, T, R1, R2, P1, P2, Q, mapl1, mapl2, mapr1, mapr2);
	cout << "已创建图像重投影映射表！" << endl;

	//视差图需要通过两幅图像共同完成
	computeDisparityImage(imageName_L, imageName_R, img1_rectified, img2_rectified, mapl1, mapl2, mapr1, mapr2, validRoi, disparity);
	cout << "视差图建立完成！" << endl;

	imshow("视差图", disparity);
	// 从三维投影获得深度映射，这里的result3DImage是指重投影矩阵 Q 实现了世界坐标系与图像像素坐标系之间的坐标转换后的世界坐标系，获取result3DImage的值
	reprojectImageTo3D(disparity, result3DImage, Q);
	imshow("视差图", disparity);
	imshow("result3DImage图", result3DImage);
	//cvtColor(result3DImage, result3DImage1, COLOR_GRAY2BGR);
	//imshow("result3DImage", result3DImage1);
	//回调函数
	setMouseCallback("视差图", onMouse);

	waitKey(0);
	//destroyAllWindows();
	return 0;
}
