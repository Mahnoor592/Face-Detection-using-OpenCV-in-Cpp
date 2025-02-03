#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>  // For controlling the number of decimal places

using namespace std;
using namespace cv;

// Base class for video processing
class VideoProcessor {
protected:
    VideoCapture video; 
    Mat frame;

public:
    VideoProcessor(int cameraIndex = 0) {
        video.open(cameraIndex);
        if (!video.isOpened()) {
            cerr << "Error: Could not open the camera." << endl;
            throw runtime_error("Error: Could not open the camera.");
        }
    }

    virtual void processFrame() = 0;

    void displayFrame(const string& windowName) {
        imshow(windowName, frame);
    }

    virtual ~VideoProcessor() {
        video.release();
        destroyAllWindows();
    }
};

class FaceDetector : public VideoProcessor {
private:
    CascadeClassifier faceCascade;
    vector<Rect> faces;
    vector<bool> facesSaved;
    double focalLength;  // Focal length of the camera
    double realFaceWidth;  // Real-world width of the face (in cm)
    VideoWriter videoWriter;  // Video writer to save the video
    int frameSkipCount;  // Number of frames to skip for speeding up the video
    int frameCounter;    // Frame counter to keep track of skipped frames

public:
    // Constructor with focal length, real face width, video file path, and speed factor
    FaceDetector(const string& cascadePath, double focalLength, double realFaceWidth, const string& videoPath, int cameraIndex = 0, int speedUpFactor = 1)
        : VideoProcessor(cameraIndex), focalLength(focalLength), realFaceWidth(realFaceWidth), frameSkipCount(speedUpFactor - 1), frameCounter(0) {
        if (!faceCascade.load(cascadePath)) {
            cerr << "Error: Could not load face cascade." << endl;
            throw runtime_error("Error: Could not load face cascade.");
        }

        // Initialize the video writer
        videoWriter.open(videoPath, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(640, 480), true);
        if (!videoWriter.isOpened()) {
            cerr << "Error: Could not open the video writer." << endl;
            throw runtime_error("Error: Could not open the video writer.");
        }
    }

    void processFrame() override {
        video.read(frame);
        if (frame.empty()) {
            cerr << "Error: Could not read a frame from the camera." << endl;
            throw runtime_error("Error: Could not read a frame from the camera.");
        }

        frameCounter++;

        // Skip frames based on the speed-up factor
        if (frameCounter % (frameSkipCount + 1) != 0) {
            return;  // Skip this frame
        }

        frameCounter = 0;  // Reset frame counter after saving a frame

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

        if (facesSaved.size() != faces.size()) {
            facesSaved.resize(faces.size(), false);
        }

        annotateFrame();

        // Write the processed frame to the video file
        videoWriter.write(frame);
    }

    string getUniqueFilename() {
        filesystem::create_directory("faces");
        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        string filename = "faces/face_" + to_string(now) + ".jpg";
        int count = 1;
        string newFilename = filename;
        while (filesystem::exists(newFilename)) {
            newFilename = "faces/face_" + to_string(now) + "_" + to_string(count++) + ".jpg";
        }
        return newFilename;
    }

    void annotateFrame() {
        string faceMessage = to_string(faces.size()) + " face" + (faces.size() > 1 ? "s" : "") + " found";

        for (size_t i = 0; i < faces.size(); ++i) {
            rectangle(frame, faces[i], Scalar(50, 50, 255), 3);
            Mat detectedFace = frame(faces[i]);

            if (!facesSaved[i]) {
                string filename = getUniqueFilename();
                imwrite(filename, detectedFace);
                facesSaved[i] = true;
            }

            string labelText = "Face " + to_string(i + 1);

            // Calculate distance (in cm)
            double distance = estimateDistance(faces[i].width);

            // Set fixed decimal points to 2
            stringstream ss;
            ss << fixed << setprecision(2) << distance;
            labelText += " Dist: " + ss.str() + " cm";

            putText(frame, labelText, faces[i].tl(), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 1);
        }

        putText(frame, faceMessage, Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 1);
    }

    // Function to estimate distance to the camera
    double estimateDistance(int faceWidthInPixels) {
        // Using the formula: Distance = (Real Face Width * Focal Length) / Face Width in Pixels
        return (realFaceWidth * focalLength) / faceWidthInPixels;
    }
};

int main() {
    try {
        // Assuming focal length = 800 (change based on your camera calibration)
        // Assuming real face width = 14 cm (average width of a human face)
        double focalLength = 800;
        double realFaceWidth = 14.0;  // in cm

        // Specify the video path to save the video inside "faces" folder
        string videoPath = "faces/output_video.avi";  // Video output file inside "faces" folder

        // Speed-up factor: 1 means no speed-up, 2 means 2x speed-up, 3 means 3x speed-up, etc.
        int speedUpFactor = 2;

        // Create FaceDetector object to process video and save it
        FaceDetector faceDetector("haarcascade_frontalface_default.xml", focalLength, realFaceWidth, videoPath, 0, speedUpFactor);

        while (true) {
            faceDetector.processFrame();
            faceDetector.displayFrame("Face Detection");

            if (waitKey(20) == 'q') {
                break;
            }
        }
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}