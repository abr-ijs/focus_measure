#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <std_msgs/Float64MultiArray.h>
#include <fstream>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>

static void showUsage(std::string name){
    std::cerr << "Usage: " << name << " <options> SOURCES\n"
              << "Options:\n"
              << "\t -h, --help\t\t\t\tShow this message\n"
              << "\t -s, --source\t\tSOURCE\t\tSpecify the source path\n"
              << "\t -d, --destination\tDESTINATION\tSpecify the destiantion path"
              << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 5){
        showUsage(argv[0]);
        return 1;
    }

    std::string source, destination;
    for (int i = 1; i < argc; i++){
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")){
            showUsage(argv[0]);
            return 0;
        } else if ((arg == "-s") || (arg == "--source")){
            if ((i + 1 < argc) && ((arg != "-d") || (arg != "--destination")) ){
                source = argv[i + 1];                
                if (boost::filesystem::extension(source) == ".avi"){                    
                } else{
                    std::cerr << "Source file type must be avi." << std::endl;
                    return 1;
                }
            }                 
        } else if ((arg == "-d") || (arg == "--destination")){
            if (i + 1 < argc){
                destination = argv[i + 1];
                if (boost::filesystem::extension(destination) == ".txt"){
                    
                 } else{
                     std::cerr << "Source file type must be txt." << std::endl;
                    return 1;
                 }
            }
        }
    }
    
    cv::VideoCapture capture(source);
    cv::Mat frame;
    std_msgs::Float64MultiArray focus_data;
    focus_data.data.resize(24);
    int counter = 0;
    //filters definition
    //1st order derivative of Brenner operator with filter2D function call
    cv::Mat brennerH, brennerV, absBrennerH, absBrennerV, sqAbsBrennerH, sqAbsBrennerV, kernBrennerH, kernBrennerV;
    cv::Scalar sumBrennerH, sumBrennerV;
    kernBrennerH = (cv::Mat_<float>(5,5)<<0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,1,0,0);
    kernBrennerV = (cv::Mat_<float>(5,5)<<0,0,0,0,0,0,0,0,0,0,0,0,-1,0,1,0,0,0,0,0,0,0,0,0,0);
    //1st order derivative of squared gradient operator with filter2D function call/*
    cv::Mat gradH, gradV, absGradH, absGradV, sqAbsGradH, sqAbsGradV, kernGradH, kernGradV;
    cv::Scalar sumGradH, sumGradV;
    kernGradH = (cv::Mat_<float>(3,3)<<0,0,0,0,-1,1,0,0,0);
    kernGradV = (cv::Mat_<float>(3,3)<<0,0,0,0,-1,0,0,1,0);
    //1st order derivative of 3x3 difference operator with filter2D function call
    cv::Mat differenceH, differenceV, absDifferenceH, absDifferenceV, sqAbsDifferenceH, sqAbsDifferenceV, kernDifferenceH, kernDifferenceV;
    cv::Scalar sumDifferenceH, sumDifferenceV;
    kernDifferenceH = (cv::Mat_<float>(3,3)<<0,0,0,-1,0,1,0,0,0);
    kernDifferenceV = (cv::Mat_<float>(3,3)<<0,-1,0,0,0,0,0,1,0);
    //1st order derivative of 3x3 Sobel operator with filter2D function call
    cv::Mat sobel3H, sobel3V, absSobel3H, absSobel3V,  sqAbsSobel3H, sqAbsSobel3V, kernSobel3H, kernSobel3V;
    cv::Scalar sumSobel3H, sumSobel3V;
    kernSobel3H = (cv::Mat_<float>(3,3)<<-1,0,1,-2,0,2,-1,0,1);
    kernSobel3V = (cv::Mat_<float>(3,3)<<-1,-2,-1,0,0,0,1,2,1);
    //1st order derivative of 3x3 Scharr operator with filter2D function call
    cv::Mat scharr3H, scharr3V, absScharr3H, absScharr3V, sqAbsScharr3H, sqAbsScharr3V, kernScharr3H, kernScharr3V;
    cv::Scalar sumScharr3H, sumScharr3V;
    kernScharr3H = (cv::Mat_<float>(3,3)<<-3,0,3,-10,0,10,-3,0,3);
    kernScharr3V = (cv::Mat_<float>(3,3)<<-3,-10,-3,0,0,0,3,10,3);
    //1st order derivative of 3x3 Roberts operator with filter2D function call
    cv::Mat roberts3H, roberts3V, absRoberts3H, absRoberts3V, sqAbsRoberts3H, sqAbsRoberts3V, kernRoberts3H, kernRoberts3V;
    cv::Scalar sumRoberts3H, sumRoberts3V;
    kernRoberts3H = (cv::Mat_<float>(3,3)<<0,0,0,0,1,0,-1,0,0);
    kernRoberts3V = (cv::Mat_<float>(3,3)<<0,0,0,0,1,0,0,0,-1);
    //1st order derivative of 3x3 Prewitt operator with filter2D function call
    cv::Mat prewitt3H, prewitt3V, absPrewitt3H, absPrewitt3V, sqAbsPrewitt3H, sqAbsPrewitt3V, kernPrewitt3H, kernPrewitt3V;
    cv::Scalar sumPrewitt3H, sumPrewitt3V;
    kernPrewitt3H = (cv::Mat_<float>(3,3)<<-1,0,1,-1,0,1,-1,0,1);
    kernPrewitt3V = (cv::Mat_<float>(3,3)<<-1,-1,-1,0,0,0,1,1,1);
    //1st order derivative of 5x5 Sobel operator with filter2D function call
    cv::Mat sobel5H, sobel5V, absSobel5H, absSobel5V, sqAbsSobel5H, sqAbsSobel5V, kernSobel5H, kernSobel5V;
    cv::Scalar sumSobel5H, sumSobel5V;
    kernSobel5H = (cv::Mat_<float>(5,5)<<-1,-2,0,2,1,-4,-8,0,8,4,6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1);
    kernSobel5V = (cv::Mat_<float>(5,5)<<-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1);
    //2nd order derivative of 3x3 Sobel operator with filter2D function call
    cv::Mat sobel3H2nd, sobel3V2nd, absSobel3H2nd, absSobel3V2nd, sqAbsSobel3H2nd, sqAbsSobel3V2nd, kernSobel3H2nd, kernSobel3V2nd;
    cv::Scalar sumSobel3H2nd, sumSobel3V2nd;
    kernSobel3H2nd = (cv::Mat_<float>(3,3)<<1,2,1,-2,-4,-2,1,2,1);
    kernSobel3V2nd = (cv::Mat_<float>(3,3)<<1,-2,1,2,-4,2,1,-2,1);
    //2nd order derivative of 5x5 Sobel operator with filter2D function call
    cv::Mat sobel5H2nd, sobel5V2nd, absSobel5H2nd, absSobel5V2nd, sqAbsSobel5H2nd, sqAbsSobel5V2nd, kernSobel5H2nd, kernSobel5V2nd;
    cv::Scalar sumSobel5H2nd, sumSobel5V2nd;
    kernSobel5H2nd = (cv::Mat_<float>(5,5)<<1,4,6,4,10,0,0,0,0,0,-2,-8,-12,-8,-2,0,0,0,0,0,1,4,6,4,1);
    kernSobel5V2nd = (cv::Mat_<float>(5,5)<<1,0,-2,0,1,4,0,-8,0,4,6,0,-12,0,6,4,0,-8,0,4,1,0,-2,0,1);
    //2nd order derivative of 5x5 Sobel operator with filter2D function call
    cv::Mat sobel32nd, sobel52nd, absSobel32nd, absSobel52nd, sqAbsSobel32nd, sqAbsSobel52nd, kernSobel32nd, kernSobel52nd;
    cv::Scalar sumSobel32nd, sumSobel52nd;
    kernSobel32nd = (cv::Mat_<float>(3,3)<<-1,0,1,0,0,0,1,0,-1);
    kernSobel52nd = (cv::Mat_<float>(5,5)<<-1,-2,0,2,1,-2,-4,0,4,2,0,0,0,0,0,2,4,0,-4,-2,1,2,0,-2,-1);
    //2nd order derivative of 5x5 Laplacian operator with filter2D function call
    cv::Mat laplacian32nd, laplacian52nd, absLaplacian32nd, absLaplacian52nd, sqAbsLaplacian32nd, sqAbsLaplacian52nd, kernLaplacian32nd, kernLaplacian52nd;
    cv::Scalar sumLaplacian32nd, sumLaplacian52nd;
    kernLaplacian32nd = (cv::Mat_<float>(3,3)<<-1,-1,-1,-1,8,-1,-1,-1,-1);
    kernLaplacian52nd = (cv::Mat_<float>(5,5)<<-1,-3,-4,-3,-1,-3,0,6,0,-3,-4,6,20,6,-4,-3,0,6,0,-3,-1,-3,-4,-3,-1);
    
    
    if ( !capture.isOpened() ){
        std::cerr << "Error when reading file " << source << std::endl;
    }

    if ( remove( destination.c_str() ) == 0 ){        
        std::cout << "File " << destination << " already exists. Deleting file." << std::endl;
    }

    std::ofstream myfile (destination.c_str(), std::ios::app);
    if (myfile.is_open()){
        std::cout << "Calculating focus measures" << std::flush;
        myfile << "BrennerH BrennerV GradH GradV DifferenceH DifferenceV Sobel3H Sobel3V Scharr3H Scharr3V Roberts3H Roberts3V Prewitt3H Prewitt3V 
                   Sobel5H Sobel5V Sobel3H2nd Sobel3V2nd Sobel5H2nd Sobel5V2nd Sobel32nd Sobel52nd Laplacian32nd Laplacian52nd" << std::endl;
        for( ; ; ){
            capture >> frame;
            if(frame.empty()){
                break;
            }
            
            // filter2D computes correlation
            cv::filter2D(frame,brennerH,CV_32F,kernBrennerH);
            cv::convertScaleAbs(brennerH,absBrennerH);
            sqAbsBrennerH = absBrennerH.mul(absBrennerH);
            cv::filter2D(frame,brennerV,CV_32F,kernBrennerV);
            cv::convertScaleAbs(brennerV,absBrennerV);
            sqAbsBrennerV = absBrennerV.mul(absBrennerV);
            sumBrennerH = cv::sum(sqAbsBrennerH);
            sumBrennerH = sumBrennerH/absBrennerH.rows/absBrennerH.cols;           
            sumBrennerV = cv::sum(sqAbsBrennerV);
            sumBrennerV = sumBrennerV/absBrennerV.rows/absBrennerV.cols;  
                
            // filter2D computes correlation
            cv::filter2D(frame,gradH,CV_32F,kernGradH);            
            cv::convertScaleAbs(gradH,absGradH);
            sqAbsGradH = absGradH.mul(absGradH);
            cv::filter2D(frame,gradV,CV_32F,kernGradV);
            cv::convertScaleAbs(gradH,absGradH);
            sqAbsGradV = absGradV.mul(absGradV);
            sumGradH = cv::sum(sqAbsGradH);
            sumGradH = sumGradH/absGradH.rows/absGradH.cols;           
            sumGradV = cv::sum(sqAbsGradV);
            sumGradV = sumGradV/absGradV.rows/absGradV.cols;  
    
            // filter2D computes correlation
            cv::filter2D(frame,differenceH,CV_32F,kernDifferenceH);
            cv::convertScaleAbs(differenceH,absDifferenceH);
            sqAbsDifferenceH = absDifferenceH.mul(absDifferenceH);
            cv::filter2D(frame,differenceV,CV_32F,kernDifferenceV);
            cv::convertScaleAbs(differenceV,absDifferenceV);
            sqAbsDifferenceV = absDifferenceV.mul(absDifferenceV);
            sumDifferenceH = cv::sum(sqAbsDifferenceH);
            sumDifferenceH = sumDifferenceH/absDifferenceH.rows/absDifferenceH.cols;           
            sumDifferenceV = cv::sum(sqAbsDifferenceV);
            sumDifferenceV = sumDifferenceV/absDifferenceV.rows/absDifferenceV.cols;
    
            // filter2D computes correlation
            cv::filter2D(frame,sobel3H,CV_32F,kernSobel3H);
            cv::convertScaleAbs(sobel3H,absSobel3H);
            sqAbsSobel3H = absSobel3H.mul(absSobel3H);
            cv::filter2D(frame,sobel3V,CV_32F,kernSobel3V);
            cv::convertScaleAbs(sobel3V,absSobel3V);            
            sqAbsSobel3V = absSobel3V.mul(absSobel3V);
            sumSobel3H = cv::sum(sqAbsSobel3H);
            sumSobel3H = sumSobel3H/absSobel3H.rows/absSobel3H.cols;           
            sumSobel3V = cv::sum(sqAbsSobel3V);
            sumSobel3V = sumSobel3V/absSobel3V.rows/absSobel3V.cols;
    
            // filter2D computes correlation
            cv::filter2D(frame,scharr3H,CV_32F,kernScharr3H);
            cv::convertScaleAbs(scharr3H,absScharr3H);
            sqAbsScharr3H = absScharr3H.mul(absScharr3H);
            cv::filter2D(frame,scharr3V,CV_32F,kernScharr3V);
            cv::convertScaleAbs(scharr3V,absScharr3V);
            sqAbsScharr3V = absScharr3V.mul(absScharr3V);
            sumScharr3H = cv::sum(sqAbsScharr3H);
            sumScharr3H = sumScharr3H/absScharr3H.rows/absScharr3H.cols;           
            sumScharr3V = cv::sum(sqAbsScharr3V);
            sumScharr3V = sumScharr3V/absScharr3V.rows/absScharr3V.cols;
    
            // filter2D computes correlation
            cv::filter2D(frame,roberts3H,CV_32F,kernRoberts3H);
            cv::convertScaleAbs(roberts3H,absRoberts3H);
            sqAbsRoberts3H = absRoberts3H.mul(absRoberts3H);
            cv::filter2D(frame,roberts3V,CV_32F,kernRoberts3V);
            cv::convertScaleAbs(roberts3V,absRoberts3V);
            sqAbsRoberts3V = absRoberts3V.mul(absRoberts3V);
            sumRoberts3H = cv::sum(sqAbsRoberts3H);
            sumRoberts3H = sumRoberts3H/absRoberts3H.rows/absRoberts3H.cols;           
            sumRoberts3V = cv::sum(sqAbsRoberts3V);
            sumRoberts3V = sumRoberts3V/absRoberts3V.rows/absRoberts3V.cols;

            // filter2D computes correlation
            cv::filter2D(frame,prewitt3H,CV_32F,kernPrewitt3H);
            cv::convertScaleAbs(prewitt3H,absPrewitt3H);
            sqAbsPrewitt3H = absPrewitt3H.mul(absPrewitt3H);
            cv::filter2D(frame,prewitt3V,CV_32F,kernPrewitt3V);
            cv::convertScaleAbs(prewitt3V,absPrewitt3V);
            sqAbsPrewitt3V = absPrewitt3H.mul(absPrewitt3V);            
            sumPrewitt3H = cv::sum(sqAbsPrewitt3H);
            sumPrewitt3H = sumPrewitt3H/absPrewitt3H.rows/absPrewitt3H.cols;           
            sumPrewitt3V = cv::sum(sqAbsPrewitt3V);
            sumPrewitt3V = sumPrewitt3V/absPrewitt3V.rows/absPrewitt3V.cols;
    
            // filter2D computes correlation
            cv::filter2D(frame,sobel5H,CV_32F,kernSobel5H);
            cv::convertScaleAbs(sobel5H,absSobel5H);
            sqAbsSobel5H = absSobel5H.mul(absSobel5H);
            cv::filter2D(frame,sobel5V,CV_32F,kernSobel5V);
            cv::convertScaleAbs(sobel5V,absSobel5V);
            sqAbsSobel5V = absSobel5V.mul(absSobel5V);            
            sumSobel5H = cv::sum(sqAbsSobel5H);
            sumSobel5H = sumSobel5H/absSobel5H.rows/absSobel5H.cols;           
            sumSobel5V = cv::sum(sqAbsSobel5V);
            sumSobel5V = sumSobel5V/absSobel5V.rows/absSobel5V.cols;
    
            // filter2D computes correlation
            cv::filter2D(frame,sobel3H2nd,CV_32F,kernSobel3H2nd);
            cv::convertScaleAbs(sobel3H2nd,absSobel3H2nd);
            sqAbsSobel3H2nd = absSobel3H2nd.mul(absSobel3H2nd);
            cv::filter2D(frame,sobel3V2nd,CV_32F,kernSobel3V2nd);
            cv::convertScaleAbs(sobel3V2nd,absSobel3V2nd);
            sqAbsSobel3V2nd = absSobel3V2nd.mul(absSobel3V2nd);            
            sumSobel3H2nd = cv::sum(sqAbsSobel3H2nd);
            sumSobel3H2nd = sumSobel3H2nd/absSobel3H2nd.rows/absSobel3H2nd.cols;           
            sumSobel3V2nd = cv::sum(sqAbsSobel3V2nd);
            sumSobel3V2nd = sumSobel3V2nd/absSobel3V2nd.rows/absSobel3V2nd.cols;
            
            // filter2D computes correlation
            cv::filter2D(frame,sobel5H2nd,CV_32F,kernSobel5H2nd);
            cv::convertScaleAbs(sobel5H2nd,absSobel5H2nd);
            sqAbsSobel5H2nd = absSobel5H2nd.mul(absSobel5H2nd);
            cv::filter2D(frame,sobel5V2nd,CV_32F,kernSobel5V2nd);
            cv::convertScaleAbs(sobel5V2nd,absSobel5V2nd);
            sqAbsSobel5V2nd = absSobel5V2nd.mul(absSobel5V2nd);
            sumSobel5H2nd = cv::sum(sqAbsSobel5H2nd);
            sumSobel5H2nd = sumSobel5H2nd/absSobel5H2nd.rows/absSobel5H2nd.cols;           
            sumSobel5V2nd = cv::sum(sqAbsSobel5V2nd);
            sumSobel5V2nd = sumSobel5V2nd/absSobel5V2nd.rows/absSobel5V2nd.cols;
                
            // filter2D computes correlation
            cv::filter2D(frame,sobel32nd,CV_32F,kernSobel32nd);
            cv::convertScaleAbs(sobel32nd,absSobel32nd);
            sqAbsSobel32nd = absSobel32nd.mul(absSobel32nd);
            cv::filter2D(frame,sobel52nd,CV_32F,kernSobel52nd);
            cv::convertScaleAbs(sobel52nd,absSobel52nd);
            sqAbsSobel52nd = absSobel52nd.mul(absSobel52nd);
            sumSobel32nd = cv::sum(sqAbsSobel32nd);
            sumSobel32nd = sumSobel32nd/absSobel32nd.rows/absSobel32nd.cols;           
            sumSobel52nd = cv::sum(sqAbsSobel52nd);
            sumSobel52nd = sumSobel52nd/absSobel52nd.rows/absSobel52nd.cols;
                
            // filter2D computes correlation
            cv::filter2D(frame,laplacian32nd,CV_32F,kernLaplacian32nd);
            cv::convertScaleAbs(laplacian32nd,absLaplacian32nd);
            sqAbsLaplacian32nd = absLaplacian32nd.mul(absLaplacian32nd);
            cv::filter2D(frame,laplacian52nd,CV_32F,kernLaplacian52nd);
            cv::convertScaleAbs(laplacian52nd,absLaplacian52nd);
            sqAbsLaplacian52nd = absLaplacian52nd.mul(absLaplacian52nd);
            sumLaplacian32nd = cv::sum(sqAbsLaplacian32nd);
            sumLaplacian32nd = sumLaplacian32nd/absLaplacian32nd.rows/absLaplacian32nd.cols;           
            sumLaplacian52nd = cv::sum(sqAbsLaplacian52nd);
            sumLaplacian52nd = sumLaplacian52nd/absLaplacian52nd.rows/absLaplacian52nd.cols;    
                                     
            myfile << sumBrennerH[0] << ", " << sumBrennerV[0] << ", "  << sumGradH[0] << ", " << sumGradV[0] << ", " << sumDifferenceH[0] << ", " 
                << sumDifferenceV[0] << ", " << sumSobel3H[0] << ", " << sumSobel3V[0] << ", " << sumScharr3H[0] << ", " << sumScharr3V[0] << ", " 
                << sumRoberts3H[0] << ", " << sumRoberts3V[0] << ", " << sumPrewitt3H[0] << ", " << sumPrewitt3V[0] << ", "  << sumSobel5H[0] << ", " 
                << sumSobel5V[0] << ", "  << sumSobel3H2nd[0] << ", " << sumSobel3V2nd[0] << ", " << sumSobel5H2nd[0] << ", " << sumSobel5V2nd[0] << ", " 
                << sumSobel32nd[0] << ", " << sumSobel52nd[0] << ", " << sumLaplacian32nd[0] << ", " << sumLaplacian52nd[0] << std::endl;

            std::cout << " ." << std::flush;
        }
        std::cout << std::endl;
        std::cout << "Focus measures succesfully written to file " << destination << std::endl;   

    } else{
        std::cerr << "Unable to open file " << destination << std::endl; 
    }

    myfile.close();
}