#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <string.h>
#include <fstream>
using namespace std;

#define GRAYSCALE 0
#define BGR 1

//luminance quantization matrix
int lum_quant[64] = {
	16, 11, 10, 16, 24, 40, 51, 61,
	12, 12, 14, 19, 26, 58, 60, 55,
	14, 13, 16, 24, 40, 57, 69, 56,
	14, 17, 22, 29, 51, 87, 80, 62,
	18, 22, 37, 56, 68, 109, 103, 77,
	24, 35, 55, 64, 81, 104, 113, 92,
	49, 64, 78, 87, 103, 121, 120, 101,
	72, 92, 95, 98, 112, 100, 103, 99};

cv::Mat luminance(8, 8, CV_32S, lum_quant);

ofstream myFile;
ofstream myFileTxt;
ifstream myFileIn;
char fileNameBin[50] = { 0 };

int around(double a)
{
	if (a >= 0)
	{
		return int(a + 0.5);
	}
	else
	{
		return int(a - 0.5);
	}

}

boolean isInside(int i, int j, Mat img) {

	if (i < img.rows && i >= 0 && j < img.cols && j >= 0)
		return true;

	return false;
}

Mat_<Vec3b> padding(Mat_<Vec3b> src) {

	int rows8 = src.rows % 8 != 0 ? ((src.rows / 8) + 1) * 8 : src.rows;
	int cols8 = src.cols % 8 != 0 ? ((src.cols / 8) + 1) * 8 : src.cols;
	Mat_<Vec3b> dst(rows8, cols8);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst(i, j) = src(i,j);
		}
	}

	for (int i = src.rows; i < rows8; i++) {
		for (int j = src.cols; j < cols8; j++) {
			dst(i, j) = 0;
		}
	}

	return dst;
}

Mat_<Vec3s> makeBlock(Mat_<Vec3s> src, int i_src, int j_src) {
	Mat_<Vec3s> block(8,8);

	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			block(i, j) = src(i + i_src, j + j_src);
		}
	}
	return block;
}

vector<int> zizag(Mat1s src) {
	vector<int> dst;
	for (int i = 0; i < src.cols + src.rows - 1; i++) {
		if (i % 2 == 1) {
			// down left
			int x = i < src.rows ? 0 : i - src.rows + 1;
			int y = i < src.rows ? i : src.rows - 1;
			while (x < src.cols && y >= 0) {
				dst.push_back(src(x++,y--));
			}
		}
		else {
			// up right
			int x = i < src.cols ? i : src.cols - 1;
			int y = i < src.cols ? 0 : i - src.cols + 1;
			while (x >= 0 && y < src.rows) {
				dst.push_back(src(x--,y++));
			}
		}
	}
	return dst;
}

Mat makeZigzagFromRle(vector<int> zigzag) {
	
	Mat1s dst(8,8);
	int dimension = 8;
	int lastValue = 63;
	int currNum = 0;
	int currDiag = 0;
	int loopFrom;
	int loopTo;
	int i;
	int row;
	int col;

	do
	{
		if (currDiag < dimension) // if doing the upper-left triangular half
		{
			loopFrom = 0;
			loopTo = currDiag;
		}
		else // doing the bottom-right triangular half
		{
			loopFrom = currDiag - dimension + 1;
			loopTo = dimension - 1;
		}

		for (i = loopFrom; i <= loopTo; i++)
		{
			if (currDiag % 2 == 0) // want to fill upwards
			{
				row = loopTo - i + loopFrom;
				col = i;
			}
			else // want to fill downwards
			{
				row = i;
				col = loopTo - i + loopFrom;
			}

			dst[row][col] = zigzag[currNum++];
		}

		currDiag++;
	} while (currDiag <= lastValue);

	return dst;
}

vector<int> rle(vector<int> src) {

	vector<int> result;

	for (int i = 0; i < src.size(); i++) {

		// Count occurrences of current character
		int count = 1;
		while (i < src.size() - 1 && src[i] == src[i + 1]) {
			count++;
			i++;
		}

		// Print character and its count
		result.push_back(src[i]);
		result.push_back(count);	
	}

	return result;
}

vector<int> decompressRle(vector<int> src) {
	vector<int> dst;
	for (int i = 0; i < src.size(); i += 2) {
		int value = src[i];
		int repeat = src[i + 1];

		for (int j = 0; j < repeat; j++) {
			dst.push_back(value);
		}
	}
	return dst;
}

Mat computeDCT(Mat1f src) {
	Mat1f dct(8, 8, CV_32FC1);

	float ci, cj, dct1, sum;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (i == 0)
				ci = 1 / sqrt(src.rows);
			else
				ci = sqrt(2) / sqrt(src.rows);
			if (j == 0)
				cj = 1 / sqrt(src.cols);
			else
				cj = sqrt(2) / sqrt(src.cols);

			// sum will temporarily store the sum of
			// cosine signals
			sum = 0;
			for (int x = 0; x < src.rows; x++) {
				for (int y = 0; y < src.cols; y++) {
					dct1 = src(x,y) *
						cos((2 * x + 1) * i * PI / (2 * src.rows)) *
						cos((2 * y + 1) * j * PI / (2 * src.cols));
					sum = sum + dct1;
				}
			}
			dct(i,j) = ci * cj * sum;
		}
	}

	return dct;
}

Mat computeIDCT(Mat1f src) {
	Mat1s idct(8, 8, CV_32S);

	double ck, cl;

	for (int m = 0;m < 8;m++)
	{
		for (int n = 0;n < 8;n++)
		{
			double temp = 0.0;
			for (int k = 0;k < 8;k++)
			{
				for (int l = 0;l < 8;l++)
				{
					if (k == 0)
					{
						ck = sqrt(1.0 / 8);
					}
					else
					{
						ck = sqrt(2.0 / 8);
					}
					if (l == 0)
					{
						cl = sqrt(1.0 / 8);
					}
					else
					{
						cl = sqrt(2.0 / 8);
					}

					temp += ck * cl * src(k,l) * cos((2 * m + 1) * k * PI / (2 * 8)) * cos((2 * n + 1) * l * PI / (2 * 8));

				}
			}
			idct(m,n) = around(temp);
		}
	}

	return idct;
}

void makeName(char src[], char name[]) {
	
	int point = 0;
	int dash = 0;

	for (int i = strlen(src) - 1; i>0; i--) {
		if (src[i] == '\\')
			dash = i;
		if (src[i] == '.')
			point = i;
		if (point != 0 && dash != 0)
			break;
	}

	if (point != 0 && dash != 0) {
		int size = point - dash - 1;		
		strcpy(name, src + dash + 1);
		name[size] = '\0';
		//cout << name << endl;		
	}		
}

void printVectorOfVectors(vector<vector<int>> zizzag_channel) {

	for (int m = 0; m < zizzag_channel.size(); m++)
	{
		for (int n = 0; n < zizzag_channel[m].size(); n++)
		{
			cout << zizzag_channel[m][n] << " ";
		}
		cout << endl;
	}
}

void writeFile(vector<vector<int>> srcVector) {

	for (int m = 0; m < srcVector.size(); m++)
	{
		//fprintf(fptr, "%d ", srcVector[m].size());
		int size = srcVector[m].size();

		myFileTxt << size << " " ;
		myFile.write(reinterpret_cast<char*> (&size), sizeof(int));

		//fwrite(&size, sizeof(int), 1, fptrBin);

		for (int n = 0; n < srcVector[m].size(); n++)
		{
			//fprintf(fptr, "%d ", srcVector[m][n]);
			//fwrite(&srcVector[m][n], sizeof(int), 1, fptrBin);

			myFileTxt << srcVector[m][n] << " ";
			myFile.write(reinterpret_cast<char*> (&srcVector[m][n]), sizeof(int));
		}
		//fprintf(fptr, "\n");

		myFileTxt << endl;

	}	
}

void openFileWrite(char fileNameBin[]) {

	char aux[50] = { 0 };
	strcpy_s(aux, fileNameBin);
	//myFile.open(strcat(fileNameBin, ".bin"), ios::out | ios::trunc | ios::binary);//);| ios::app
	myFile.open(strcat(aux, ".bin"), ios::out | ios::trunc | ios::binary);
}

void openFileRead(char fileNameBin[]) {

	myFileIn.open(fileNameBin, ios::binary);
}

vector<vector<int>> readRLE() {

	vector<vector<int>> rle3channels;
	int size, x;

	if (myFileIn.is_open()) {

		for (int i = 0; i < 3; i++) {
			vector<int> rle;
			myFileIn.read((char*)(&size), sizeof(int));
			//cout << size << endl;
			for (int j = 0; j < size; j++) {
				myFileIn.read(reinterpret_cast<char*>(&x), sizeof(int));
				rle.push_back(x);
				//cout << x << " " ;
			}

			rle3channels.push_back(rle);
			//cout << endl;
		}
		return rle3channels;
	}
	else
	{
		cout << "Could not read file ";
		return rle3channels;
	}
}

Mat_<Vec3s> putBlock(Mat_<Vec3s> src, Mat_<Vec3s> block, int row, int col) {

	for (int i = row; i < row + 8; i++) {
		for (int j = col; j < col + 8; j++) {
			if(isInside( i, j, src) && isInside(row % 8, col % 8, block))
			src(i, j) = block(row % 8, col % 8);
		}
	}

	return src;
}

void decompress() {

	char aux1[50] = { 0 };
	strcpy_s(aux1, fileNameBin);
	
	openFileRead(strcat(aux1, ".bin" ));//openFileRead(fileNameBin);
	//cout << fileNameBin << endl;
	int originalW = 0, originalH = 0, W8 = 0, H8 = 0;

	bool print = false;
	
	int y = 0;
	if (myFileIn.is_open()) {
		
		myFileIn.read((char*)(&originalH), sizeof(int));
		myFileIn.read(reinterpret_cast<char*>(&originalW), sizeof(int));
		myFileIn.read(reinterpret_cast<char*>(&H8), sizeof(int));
		myFileIn.read(reinterpret_cast<char*>(&W8), sizeof(int));
		//cout << originalH << " " << originalW  <<  " " << H8 << " " << W8 << endl;
		
		Mat_<Vec3s> signedMatrix(H8, W8);
		Mat_<Vec3b> compressedImg(originalH, originalW);

		int blocksCol = (W8 / 8);
		int blocksRow = (H8 / 8);
		int totalSquares = blocksCol * blocksRow;
		int row = 0, col = 0;

		//cout << blocksCol << " " << blocksRow << endl;

		for (int i = 0; i < totalSquares; i++) {
			vector<vector<int>> crtRLE = readRLE();
			vector<int> decompressedRle;
			
			Mat block(8, 8, CV_32SC3);
			vector<Mat> channels;
			split(block, channels);

			for (int j = 0; j < crtRLE.size(); j++) {

				decompressedRle = decompressRle(crtRLE[j]);
				Mat1s blockCrt = makeZigzagFromRle(decompressedRle);				
				Mat1s blockMul;
				multiply(blockCrt, luminance, blockMul);
				channels[j] = computeIDCT(blockMul);
				channels[j].convertTo(channels[j], CV_32S);
	
			}

			merge(channels, block);

			if ( (col + 8) == W8 ) {	
				row += 8;			
			}
			
			col = i % blocksCol * 8;
			signedMatrix = putBlock(signedMatrix, block, row, col);

			/*if (i <= 4){
				//cout << endl << " block dec i = " << i << endl << channels[0] << endl << channels[1] << endl << channels[2] << endl;
				//cout << endl << " block dec i = " << i << endl << block << endl;
				
				cout << endl << " signed matrix = " << i << endl;
				for (int i = 0; i < 10; i++)
					cout << signedMatrix(0,i) << " ";
			}*/

		}

		//convert to unsigned
		add(signedMatrix, Scalar(128, 128, 128), compressedImg); //cout << compressedImg;
		imshow("compressed image YCrCb", compressedImg);

		Mat_<Vec3b> finalImg;
		cv::cvtColor(compressedImg, finalImg, cv::COLOR_YCrCb2BGR);
		imshow("compressed image BGR", finalImg);

		cout << endl << " final img "  << endl;
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				cout << finalImg(i, j) << " ";			
			}cout << endl;
		}

		char aux2[50] = { 0 };
		strcpy_s(aux2, fileNameBin);

		imwrite( strcat(aux2, ".jpg"), finalImg);
	}
	else
	{
		cout << "Could not read file ";
	}
	
	myFileIn.close();
}

void closeFile() {

	myFile.close();
	myFileTxt.close();
}

void compress(Mat_<Vec3b> src) {

	Mat_<Vec3b> dst;
	Mat_<Vec3b> ycrcb;
	Mat_<Vec3s> signed128;

	luminance.convertTo(luminance, CV_32FC1);
	cout << luminance << endl;

	//if image does not have w and h multiple of 8, pad the matrix with 0
	if (src.cols % 8 != 0 || src.rows % 8 != 0) {
		dst = padding(src);
	}
	else{
		src.copyTo(dst);
	}

	myFile.write(reinterpret_cast<char*> (&src.rows), sizeof(int));
	myFile.write(reinterpret_cast<char*> (&src.cols), sizeof(int));
	myFile.write(reinterpret_cast<char*> (&dst.rows), sizeof(int));
	myFile.write(reinterpret_cast<char*> (&dst.cols), sizeof(int));

	cout << endl << " initial image = " << endl;
	for (int i = 0; i < 10; i++){
		for(int j=0; j < 10; j++){
			cout << src(i, j) << " ";
		}cout << endl;
	}

	//convert to YCrCb
	cv::cvtColor(dst, ycrcb, cv::COLOR_BGR2YCrCb);imshow("YCrCb", ycrcb);

	//convert to signed
	subtract(ycrcb, Scalar(128, 128, 128), signed128);//cout << signed128;
	
	//divide into 8x8 blocks
	for (int i = 0; i < signed128.rows; i += 8)
	{
		for (int j = 0; j < signed128.cols; j += 8)
		{
			Mat_<Vec3s> block(8, 8);
			block = makeBlock(signed128, i, j);

			vector<Mat> channels;
			split(block, channels);

			vector<Mat> DCT;
			vector<Mat> DCTMatrix;
			vector<vector<int>> zizzag_channel;
			vector<vector<int>> rle_zigzag;
			
			//if (i == 0 && j<=4*8 )
				//cout << endl << " block init (i,j)" << i << "," << j << endl << channels[0] << endl << channels[1] << endl << channels[2] << endl;
		
			for (int k = 0; k < 3; k++) {
				DCT.push_back(Mat(8, 8, CV_32FC1));
				DCTMatrix.push_back(Mat(8, 8, CV_32FC1));
				channels[k].convertTo(channels[k], CV_32FC1);			
				DCTMatrix[k] = computeDCT(channels[k]);

				//if (i == 0 && j <= 8)
					//cout << endl << " block init (i,j)" << i << "," << j << endl << channels[k] << endl;
				
				divide(DCTMatrix[k], luminance, channels[k]);
				channels[k].convertTo(channels[k], CV_32S);
				vector<int> crtZigZag = zizag(channels[k]);
				zizzag_channel.push_back(crtZigZag);
				rle_zigzag.push_back(rle(crtZigZag));
			}

			writeFile(rle_zigzag);
		}
	}

	closeFile();
}

void mouseClick()
{
	Mat_<Vec3b> src;
	
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname);
		
		char fileName[50] = { 0 };
		
		makeName(fname, fileName);
		makeName(fname, fileNameBin);

		openFileWrite(fileNameBin);

		imshow("Original", src);
		compress(src);
		
		decompress();
	}
}

int main() {

	mouseClick();
	
	waitKey(0);

	return 0;
}

